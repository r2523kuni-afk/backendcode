"""
CGPA Tracker API  —  v2  (+ SQLite database)
=============================================
New endpoints added under /api/v1/students/:

  POST   /students                        — create student profile
  GET    /students                        — list all students
  GET    /students/{id}                   — get student
  PATCH  /students/{id}                   — update student
  DELETE /students/{id}                   — delete student

  POST   /students/{id}/semesters         — add/update a semester record
  GET    /students/{id}/semesters         — list semester records
  DELETE /students/{id}/semesters/{sem}   — delete one semester

  POST   /students/{id}/plans             — save a CGPA plan
  GET    /students/{id}/plans             — list saved plans
  DELETE /plans/{plan_id}                 — delete a plan

  GET    /students/{id}/history           — calculation history for student
  GET    /history                         — all recent calculation history

All original planning endpoints (/calculate-plan, /course-plan, etc.) now
accept an optional  student_id  query parameter to auto-log results.
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal, ROUND_HALF_UP, getcontext
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Database layer ─────────────────────────────────────────────────────────
from database import (
    init_db,
    create_student, get_student, get_student_by_email,
    update_student, delete_student, list_students,
    upsert_semester_record, get_semester_records, delete_semester_record,
    save_plan, get_plan, list_plans, delete_plan,
    log_calculation, get_calculation_history,
)

getcontext().prec = 28

try:
    from scipy.optimize import linprog, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CGPA Tracker API",
    version="2.1.0",
    description="Academic planning engine with SQLite persistence.",
    openapi_tags=[
        {"name": "setup",    "description": "Grading system configuration"},
        {"name": "students", "description": "Student profiles & academic records"},
        {"name": "planning", "description": "Semester & course planning"},
        {"name": "analysis", "description": "Sensitivity, simulation, health"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Create DB tables on first run."""
    init_db()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MODELS  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

class GradeEntry(BaseModel):
    grade:       str
    points:      float
    min_percent: Optional[float] = None


class GradingSystem(BaseModel):
    scale_max:       float = Field(..., gt=0)
    passing_points:  float = Field(..., ge=0)
    grade_map:       list[GradeEntry]
    credit_range:    list[int] = Field(default=[15, 25])
    decimal_places:  int       = Field(default=2, ge=0, le=4)

    @field_validator("grade_map")
    @classmethod
    def grade_map_nonempty(cls, v):
        if not v:
            raise ValueError("grade_map cannot be empty")
        return v

    @model_validator(mode="after")
    def cross_validate(self):
        if self.passing_points > self.scale_max:
            raise ValueError("passing_points cannot exceed scale_max")
        if max(g.points for g in self.grade_map) != self.scale_max:
            raise ValueError(f"Highest grade point must equal scale_max ({self.scale_max})")
        if len(self.credit_range) != 2 or self.credit_range[0] <= 0 or self.credit_range[1] < self.credit_range[0]:
            raise ValueError("credit_range must be [min, max] with min > 0")
        return self


class SemesterRecord(BaseModel):
    semester_number: int
    sgpa:            float
    credits:         int = Field(..., gt=0)


class AcademicState(BaseModel):
    semester_history:      Optional[list[SemesterRecord]] = None
    current_cgpa:          Optional[float]                = None
    total_credits_earned:  Optional[int]                  = Field(default=None, gt=0)

    @model_validator(mode="after")
    def exactly_one_mode(self):
        has_history = bool(self.semester_history)
        has_summary = self.current_cgpa is not None and self.total_credits_earned is not None
        if has_history and has_summary:
            raise ValueError("Supply either semester_history OR (current_cgpa + total_credits_earned), not both.")
        return self


# ══════════════════════════════════════════════════════════════════════════════
# DECIMAL MATH HELPERS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def D(x) -> Decimal:
    return Decimal(str(x))

def weighted_cgpa(records: list[dict]) -> Decimal:
    tw = sum(D(r["sgpa"]) * D(r["credits"]) for r in records)
    tc = sum(D(r["credits"]) for r in records)
    return tw / tc if tc else D("0")

def extract_earned(state: AcademicState) -> tuple[Decimal, int]:
    if state.semester_history:
        records = [{"sgpa": s.sgpa, "credits": s.credits} for s in state.semester_history]
        return weighted_cgpa(records), sum(r["credits"] for r in records)
    if state.current_cgpa is not None:
        return D(state.current_cgpa), state.total_credits_earned
    return D("0"), 0

def required_uniform_sgpa(curr_cgpa, curr_credits, target, future_credits):
    if future_credits == 0:
        return D("0")
    total = curr_credits + future_credits
    return (target * total - curr_cgpa * curr_credits) / future_credits

def round_dp(val: Decimal, dp: int) -> float:
    quantize_str = D("0." + "0" * dp) if dp > 0 else D("1")
    return float(val.quantize(quantize_str, rounding=ROUND_HALF_UP))

def nearest_grade(sgpa: float, grade_map: list[GradeEntry]) -> str:
    sorted_grades = sorted(grade_map, key=lambda g: g.points, reverse=True)
    for g in sorted_grades:
        if sgpa >= g.points:
            return g.grade
    return sorted_grades[-1].grade

def infer_trend(history: list[SemesterRecord]) -> str:
    if not history or len(history) < 2:
        return "stable"
    scores = [s.sgpa for s in sorted(history, key=lambda s: s.semester_number)]
    diffs  = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
    avg_diff = statistics.mean(diffs)
    variance = statistics.variance(scores) if len(scores) > 1 else 0
    if variance > 1.5:      return "volatile"
    if avg_diff > 0.15:     return "improving"
    if avg_diff < -0.15:    return "declining"
    return "stable"

def compute_max_achievable(curr_cgpa, curr_credits, future_credits, scale_max):
    total = curr_credits + future_credits
    return (curr_cgpa * curr_credits + scale_max * future_credits) / total

def feasibility_report(req, scale_max, passing, target, curr_cgpa, max_achievable, dp):
    if curr_cgpa >= target:
        return {"feasible": True, "already_achieved": True, "risk_level": "none",
                "message": "You have already met or exceeded your target CGPA."}
    if req > scale_max:
        return {"feasible": False, "already_achieved": False, "risk_level": "impossible",
                "message": (f"Mathematically impossible. Even scoring {scale_max} every semester "
                            f"yields at most {round_dp(max_achievable, dp)}."),
                "max_achievable_cgpa": round_dp(max_achievable, dp)}
    gap   = float(req - passing)
    span  = float(scale_max - passing)
    ratio = gap / span if span > 0 else 0
    if ratio > 0.85: risk, msg = "extreme",  f"Requires near-perfect performance ({round_dp(req, dp+2)})."
    elif ratio > 0.65: risk, msg = "high",   f"Demanding but achievable ({round_dp(req, dp+2)} SGPA)."
    elif ratio > 0.35: risk, msg = "moderate", f"Realistic goal ({round_dp(req, dp+2)} SGPA per semester)."
    else:              risk, msg = "low",     f"Comfortable target. Good buffer available."
    return {"feasible": True, "already_achieved": False, "risk_level": risk,
            "required_uniform_sgpa": round_dp(req, dp+2), "message": msg}

def quality_points_breakdown(curr_cgpa, curr_credits, target, future_credits):
    total = curr_credits + future_credits
    earned_qp  = curr_cgpa * curr_credits
    needed_qp  = target * total
    remaining  = needed_qp - earned_qp
    return {
        "current_quality_points":          round(float(earned_qp), 2),
        "required_total_quality_points":   round(float(needed_qp), 2),
        "remaining_quality_points_needed": round(float(remaining), 2),
        "future_credits":                  future_credits,
        "total_credits_at_graduation":     total,
        "quality_points_per_future_credit_needed":
            round(float(remaining / future_credits), 4) if future_credits else 0,
    }

def generate_recommendations(risk_level, trend, pattern, remaining_sems, req_uniform, scale_max, passing):
    recs = []
    span = scale_max - passing
    if risk_level == "impossible":
        recs.append("Target is out of reach. Consider extending your program or lowering the target.")
        return recs
    if risk_level in ("extreme", "high"):
        recs.append(f"Required SGPA of {req_uniform:.2f} leaves almost no room for error.")
        if remaining_sems <= 2:
            recs.append("With few semesters left, prioritize high-credit courses for maximum impact.")
        else:
            recs.append("Use the front-loaded strategy: perform at peak now while you have more semesters as buffer.")
    if trend == "declining":
        recs.append("Your SGPA has been declining. Investigate the cause before committing to an aggressive target.")
    elif trend == "improving":
        recs.append("Your SGPA trend is positive. Stay consistent.")
    elif trend == "volatile":
        recs.append("Volatile performance history increases risk. A stepped-up strategy helps build consistency.")
    if remaining_sems >= 4:
        recs.append("You have ample time. Small consistent improvements are more sustainable than last-minute surges.")
    return recs or ["Your plan looks balanced. Stay consistent and revisit each semester."]


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class EffortPattern(str, Enum):
    uniform      = "uniform"
    front_loaded = "front_loaded"
    back_loaded  = "back_loaded"
    stepped_up   = "stepped_up"
    stepped_down = "stepped_down"
    custom       = "custom"


class SemesterConstraint(BaseModel):
    semester_index: int           = Field(..., ge=0)
    max_sgpa:       Optional[float] = None
    min_sgpa:       Optional[float] = None
    label:          Optional[str]   = None


def _align_constraints(n, constraints):
    aligned = [None] * n
    for c in constraints:
        if 0 <= c.semester_index < n:
            aligned[c.semester_index] = c
    return aligned

def _fallback_uniform(curr_cgpa, curr_credits, target, credits, scale_max, passing, constraints):
    req = (target * (curr_credits + sum(credits)) - curr_cgpa * curr_credits) / sum(credits)
    req = max(passing, min(scale_max, req))
    aligned = _align_constraints(len(credits), constraints)
    return [max(c.min_sgpa if c and c.min_sgpa else passing,
                min(c.max_sgpa if c and c.max_sgpa else scale_max, req))
            for i, c in enumerate(aligned)]

def optimize_plan_scipy(curr_cgpa, curr_credits, target, credits_per_sem,
                         scale_max, passing, pattern, constraints, desired_buffer=0.0):
    n = len(credits_per_sem)
    future_credits = sum(credits_per_sem)
    lower = [max(passing, c.min_sgpa) if c and c.min_sgpa else passing
             for c in _align_constraints(n, constraints)]
    upper = [min(scale_max, c.max_sgpa) if c and c.max_sgpa else scale_max
             for c in _align_constraints(n, constraints)]
    rhs   = target * (curr_credits + future_credits) - curr_cgpa * curr_credits

    if pattern == EffortPattern.uniform:
        def obj(s):
            m = sum(s) / n
            return sum((x - m)**2 for x in s)
    elif pattern == EffortPattern.front_loaded:
        def obj(s): return sum(s[i] * (n - i) for i in range(n))
    elif pattern == EffortPattern.back_loaded:
        def obj(s): return sum(s[i] * (i + 1) for i in range(n))
    else:
        def obj(s):
            m = sum(s) / n
            return sum((x - m)**2 for x in s)

    from scipy.optimize import minimize
    constraints_scipy = [
        {"type": "ineq", "fun": lambda s: sum(s[i] * credits_per_sem[i] for i in range(n)) - rhs}
    ]
    bounds = [(lower[i], upper[i]) for i in range(n)]
    x0 = [min(upper[i], max(lower[i], rhs / future_credits)) for i in range(n)]
    result = minimize(obj, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints_scipy, options={"ftol": 1e-10, "maxiter": 1000})
    if result.success:
        return [max(lower[i], min(upper[i], result.x[i])) for i in range(n)], "scipy_slsqp"
    return _fallback_uniform(curr_cgpa, curr_credits, target, credits_per_sem,
                              scale_max, passing, constraints), "fallback_uniform"

def build_plan_from_sgpas(sgpas, credits_per_sem, curr_cgpa, curr_credits, dp, grade_map, sem_constraints):
    aligned = _align_constraints(len(sgpas), sem_constraints)
    rw, rc  = float(curr_cgpa) * curr_credits, curr_credits
    plan = []
    for i, (sgpa, credits) in enumerate(zip(sgpas, credits_per_sem)):
        rw += sgpa * credits
        rc += credits
        c   = aligned[i]
        plan.append({
            "semester":              i + 1,
            "target_sgpa":          round(sgpa, dp + 2),
            "credits":              credits,
            "projected_cgpa_after": round(rw / rc, dp),
            "nearest_grade":        nearest_grade(sgpa, grade_map),
            "constraint_label":     c.label if c else None,
            "is_constrained":       c is not None,
        })
    return plan


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS FOR PLANNING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    grading_system:                 GradingSystem
    academic_state:                 AcademicState
    target_cgpa:                    float
    remaining_semesters:            int = Field(..., gt=0)
    credits_per_remaining_semester: list[int]
    effort_pattern:                 EffortPattern = EffortPattern.uniform
    semester_constraints:           list[SemesterConstraint] = []
    desired_buffer:                 float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_credits_and_target(self):
        if len(self.credits_per_remaining_semester) != self.remaining_semesters:
            raise ValueError("credits_per_remaining_semester length must equal remaining_semesters")
        if any(c <= 0 for c in self.credits_per_remaining_semester):
            raise ValueError("All semester credit values must be positive")
        gs = self.grading_system
        if not (gs.passing_points <= self.target_cgpa <= gs.scale_max):
            raise ValueError(f"target_cgpa must be in [{gs.passing_points}, {gs.scale_max}]")
        return self


class CourseEntry(BaseModel):
    name:           str
    credits:        int = Field(..., gt=0)
    expected_grade: str


class CoursePlanRequest(BaseModel):
    grading_system: GradingSystem
    academic_state: AcademicState
    target_sgpa:    float
    courses:        list[CourseEntry] = Field(..., min_length=1)


class SensitivityRequest(BaseModel):
    grading_system:    GradingSystem
    academic_state:    AcademicState
    target_cgpa:       float
    future_credits:    int = Field(..., gt=0)
    probe_sgpa_values: Optional[list[float]] = None


class SimulateRequest(BaseModel):
    grading_system:    GradingSystem
    academic_state:    AcademicState
    planned_semesters: list[SemesterRecord]


class ScenarioItem(BaseModel):
    target_cgpa: float
    label:       Optional[str] = None


class ScenarioCompareRequest(BaseModel):
    grading_system:                 GradingSystem
    academic_state:                 AcademicState
    scenarios:                      list[ScenarioItem] = Field(..., min_length=2, max_length=8)
    remaining_semesters:            int = Field(..., gt=0)
    credits_per_remaining_semester: list[int]

    @model_validator(mode="after")
    def credits_match(self):
        if len(self.credits_per_remaining_semester) != self.remaining_semesters:
            raise ValueError("credits_per_remaining_semester length must equal remaining_semesters")
        return self


class AcademicHealthRequest(BaseModel):
    grading_system: GradingSystem
    academic_state: AcademicState


# ══════════════════════════════════════════════════════════════════════════════
# NEW — DB REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class StudentCreate(BaseModel):
    name:             str
    email:            str
    university:       Optional[str] = None
    program:          Optional[str] = None
    total_semesters:  int           = 8
    grading_system:   Optional[dict] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Arjun Sharma",
                "email": "arjun@example.com",
                "university": "VIT Vellore",
                "program": "B.Tech CSE",
                "total_semesters": 8,
            }
        }
    }


class StudentUpdate(BaseModel):
    name:            Optional[str]  = None
    university:      Optional[str]  = None
    program:         Optional[str]  = None
    total_semesters: Optional[int]  = None
    grading_system:  Optional[dict] = None


class SemesterRecordCreate(BaseModel):
    semester_number: int   = Field(..., ge=1)
    sgpa:            float
    credits:         int   = Field(..., gt=0)
    notes:           Optional[str] = None


class SavePlanRequest(BaseModel):
    label:          str
    target_cgpa:    float
    effort_pattern: str
    plan_data:      dict


# ══════════════════════════════════════════════════════════════════════════════
# SETUP ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/grading-presets", tags=["setup"])
def grading_presets():
    return {
        "presets": [
            {
                "name": "10-Point Scale (India)",
                "scale_max": 10.0, "passing_points": 4.0,
                "decimal_places": 2, "credit_range": [15, 30],
                "grade_map": [
                    {"grade": "O",  "points": 10.0, "min_percent": 90},
                    {"grade": "A+", "points": 9.0,  "min_percent": 80},
                    {"grade": "A",  "points": 8.0,  "min_percent": 70},
                    {"grade": "B+", "points": 7.0,  "min_percent": 60},
                    {"grade": "B",  "points": 6.0,  "min_percent": 55},
                    {"grade": "C",  "points": 5.0,  "min_percent": 50},
                    {"grade": "P",  "points": 4.0,  "min_percent": 45},
                    {"grade": "F",  "points": 0.0,  "min_percent": 0},
                ],
            },
            {
                "name": "4-Point Scale (US GPA)",
                "scale_max": 4.0, "passing_points": 1.0,
                "decimal_places": 2, "credit_range": [12, 21],
                "grade_map": [
                    {"grade": "A",  "points": 4.0}, {"grade": "A-", "points": 3.7},
                    {"grade": "B+", "points": 3.3}, {"grade": "B",  "points": 3.0},
                    {"grade": "B-", "points": 2.7}, {"grade": "C+", "points": 2.3},
                    {"grade": "C",  "points": 2.0}, {"grade": "D",  "points": 1.0},
                    {"grade": "F",  "points": 0.0},
                ],
            },
        ]
    }


@app.post("/api/v1/validate-grading-system", tags=["setup"])
def validate_grading_system(gs: GradingSystem):
    sorted_grades = sorted(gs.grade_map, key=lambda g: g.points, reverse=True)
    return {
        "valid": True,
        "normalized_grade_map": [g.model_dump() for g in sorted_grades],
        "scale_max": gs.scale_max,
        "passing_points": gs.passing_points,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/students", tags=["students"], status_code=201)
def create_student_endpoint(body: StudentCreate):
    """Create a new student profile."""
    if get_student_by_email(body.email):
        raise HTTPException(409, f"Student with email '{body.email}' already exists.")
    return create_student(
        name=body.name, email=body.email,
        university=body.university, program=body.program,
        total_semesters=body.total_semesters,
        grading_system=body.grading_system,
    )


@app.get("/api/v1/students", tags=["students"])
def list_students_endpoint():
    """List all student profiles."""
    return {"students": list_students()}


@app.get("/api/v1/students/{student_id}", tags=["students"])
def get_student_endpoint(student_id: int):
    """Get a single student profile."""
    s = get_student(student_id)
    if not s:
        raise HTTPException(404, f"Student {student_id} not found.")
    return s


@app.patch("/api/v1/students/{student_id}", tags=["students"])
def update_student_endpoint(student_id: int, body: StudentUpdate):
    """Update student profile fields."""
    s = get_student(student_id)
    if not s:
        raise HTTPException(404, f"Student {student_id} not found.")
    return update_student(student_id, **body.model_dump(exclude_none=True))


@app.delete("/api/v1/students/{student_id}", tags=["students"])
def delete_student_endpoint(student_id: int):
    """Delete a student and all their data (cascade)."""
    if not delete_student(student_id):
        raise HTTPException(404, f"Student {student_id} not found.")
    return {"deleted": True, "student_id": student_id}


# ── Semester Records ─────────────────────────────────────────────────────────

@app.post("/api/v1/students/{student_id}/semesters", tags=["students"], status_code=201)
def add_semester_record(student_id: int, body: SemesterRecordCreate):
    """Add or update a semester record for a student."""
    if not get_student(student_id):
        raise HTTPException(404, f"Student {student_id} not found.")
    records = upsert_semester_record(
        student_id, body.semester_number, body.sgpa, body.credits, body.notes
    )
    return {"student_id": student_id, "semester_records": records}


@app.get("/api/v1/students/{student_id}/semesters", tags=["students"])
def list_semester_records(student_id: int):
    """Get all semester records for a student."""
    if not get_student(student_id):
        raise HTTPException(404, f"Student {student_id} not found.")
    records = get_semester_records(student_id)
    # Also compute live CGPA from history
    if records:
        total_w = sum(r["sgpa"] * r["credits"] for r in records)
        total_c = sum(r["credits"] for r in records)
        cgpa    = round(total_w / total_c, 2)
    else:
        cgpa = None
    return {"student_id": student_id, "current_cgpa": cgpa, "semester_records": records}


@app.delete("/api/v1/students/{student_id}/semesters/{semester_number}", tags=["students"])
def remove_semester_record(student_id: int, semester_number: int):
    """Delete a specific semester record."""
    if not delete_semester_record(student_id, semester_number):
        raise HTTPException(404, "Semester record not found.")
    return {"deleted": True}


# ── Saved Plans ──────────────────────────────────────────────────────────────

@app.post("/api/v1/students/{student_id}/plans", tags=["students"], status_code=201)
def save_plan_endpoint(student_id: int, body: SavePlanRequest):
    """Save a CGPA plan for a student."""
    if not get_student(student_id):
        raise HTTPException(404, f"Student {student_id} not found.")
    return save_plan(
        student_id, body.label, body.target_cgpa,
        body.effort_pattern, body.plan_data,
    )


@app.get("/api/v1/students/{student_id}/plans", tags=["students"])
def list_saved_plans(student_id: int):
    """List all saved plans for a student."""
    if not get_student(student_id):
        raise HTTPException(404, f"Student {student_id} not found.")
    return {"student_id": student_id, "plans": list_plans(student_id)}


@app.delete("/api/v1/plans/{plan_id}", tags=["students"])
def delete_plan_endpoint(plan_id: int):
    """Delete a saved plan."""
    if not delete_plan(plan_id):
        raise HTTPException(404, f"Plan {plan_id} not found.")
    return {"deleted": True, "plan_id": plan_id}


# ── Calculation History ──────────────────────────────────────────────────────

@app.get("/api/v1/students/{student_id}/history", tags=["students"])
def student_history(student_id: int, limit: int = Query(default=20, le=100)):
    """Get recent calculation history for a student."""
    if not get_student(student_id):
        raise HTTPException(404, f"Student {student_id} not found.")
    return {"student_id": student_id,
            "history": get_calculation_history(student_id=student_id, limit=limit)}


@app.get("/api/v1/history", tags=["students"])
def all_history(limit: int = Query(default=50, le=200)):
    """Get all recent calculation history (admin view)."""
    return {"history": get_calculation_history(limit=limit)}


# ══════════════════════════════════════════════════════════════════════════════
# PLANNING ENDPOINTS  (original logic, + optional student_id logging)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/calculate-plan", tags=["planning"])
def calculate_plan(req: PlanRequest,
                   student_id: Optional[int] = Query(default=None,
                       description="If provided, result is auto-logged to history")):
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)

    if curr_credits > 0 and not (D("0") <= curr_cgpa <= D(gs.scale_max)):
        raise HTTPException(422, f"current_cgpa outside [0, {gs.scale_max}]")

    future_credits = sum(req.credits_per_remaining_semester)
    target    = D(req.target_cgpa)
    scale_max = D(gs.scale_max)
    passing   = D(gs.passing_points)

    req_uniform = required_uniform_sgpa(curr_cgpa, curr_credits, target, future_credits)
    max_ach     = compute_max_achievable(curr_cgpa, curr_credits, future_credits, scale_max)
    trend       = infer_trend(req.academic_state.semester_history or [])
    feas        = feasibility_report(req_uniform, scale_max, passing, target, curr_cgpa, max_ach, dp)
    qp          = quality_points_breakdown(curr_cgpa, curr_credits, target, future_credits)

    if SCIPY_AVAILABLE and feas["feasible"] and not feas.get("already_achieved"):
        sgpas, method = optimize_plan_scipy(
            float(curr_cgpa), curr_credits, float(target + D(req.desired_buffer)),
            req.credits_per_remaining_semester,
            float(scale_max), float(passing),
            req.effort_pattern, req.semester_constraints, req.desired_buffer,
        )
    elif feas.get("already_achieved"):
        sgpas, method = [float(curr_cgpa)] * req.remaining_semesters, "already_achieved"
    else:
        sgpas = _fallback_uniform(
            float(curr_cgpa), curr_credits, float(target),
            req.credits_per_remaining_semester,
            float(scale_max), float(passing), req.semester_constraints,
        )
        method = "fallback_uniform"

    plan = build_plan_from_sgpas(
        sgpas, req.credits_per_remaining_semester,
        curr_cgpa, curr_credits, dp, gs.grade_map, req.semester_constraints,
    )

    result = {
        "summary": {
            "current_cgpa":                round_dp(curr_cgpa, dp),
            "current_credits":             curr_credits,
            "target_cgpa":                 req.target_cgpa,
            "desired_buffer":              req.desired_buffer,
            "effective_target":            round(req.target_cgpa + req.desired_buffer, dp),
            "remaining_semesters":         req.remaining_semesters,
            "future_credits":              future_credits,
            "total_credits_at_graduation": curr_credits + future_credits,
            "max_achievable_cgpa":         round_dp(max_ach, dp),
            "performance_trend":           trend,
            "optimization_method":         method,
            "scipy_used":                  SCIPY_AVAILABLE,
        },
        "feasibility":              feas,
        "quality_points_breakdown": qp,
        "plan":                     plan,
        "trajectory": [{"semester": r["semester"], "projected_cgpa": r["projected_cgpa_after"],
                         "target_sgpa": r["target_sgpa"]} for r in plan],
        "recommendations": generate_recommendations(
            feas["risk_level"], trend, req.effort_pattern,
            req.remaining_semesters, float(req_uniform), float(scale_max), float(passing),
        ),
    }

    # Auto-log if student_id supplied
    if student_id:
        log_calculation("/api/v1/calculate-plan", req.model_dump(), result, student_id)

    return result


@app.post("/api/v1/course-plan", tags=["planning"])
def course_plan(req: CoursePlanRequest,
                student_id: Optional[int] = Query(default=None)):
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)
    grade_to_points = {g.grade: g.points for g in gs.grade_map}
    sorted_grades   = sorted(gs.grade_map, key=lambda g: g.points, reverse=True)

    course_details = []
    total_credits  = 0
    total_weighted = D("0")

    for c in req.courses:
        if c.expected_grade not in grade_to_points:
            raise HTTPException(422, f"Grade '{c.expected_grade}' not in grade_map.")
        pts = grade_to_points[c.expected_grade]
        total_weighted += D(pts) * D(c.credits)
        total_credits  += c.credits
        next_grade = next((g for g in sorted_grades if g.points > pts), None)
        impact = round((next_grade.points - pts) * c.credits / total_credits, dp+2) if next_grade else None
        course_details.append({
            "name": c.name, "credits": c.credits,
            "expected_grade": c.expected_grade, "grade_points": pts,
            "weighted_contribution": round(float(D(pts) * D(c.credits) / D(total_credits)), dp+2),
            "next_grade": next_grade.grade if next_grade else None,
            "sgpa_gain_if_upgraded": impact,
        })

    projected_sgpa = float(total_weighted / D(total_credits))
    new_cgpa = (float(curr_cgpa) * curr_credits + projected_sgpa * total_credits) / (curr_credits + total_credits)
    gap = req.target_sgpa - projected_sgpa
    sensitivity_ranked = sorted(
        [c for c in course_details if c["sgpa_gain_if_upgraded"] is not None],
        key=lambda c: c["sgpa_gain_if_upgraded"], reverse=True,
    )

    result = {
        "semester_summary": {
            "total_credits": total_credits, "projected_sgpa": round(projected_sgpa, dp),
            "target_sgpa": req.target_sgpa, "sgpa_gap": round(gap, dp+2),
            "meets_target": projected_sgpa >= req.target_sgpa,
            "new_cgpa_if_achieved": round(new_cgpa, dp),
        },
        "courses": course_details,
        "sensitivity_ranking": sensitivity_ranked,
        "advice": (
            f"Upgrade '{sensitivity_ranked[0]['name']}' from "
            f"{sensitivity_ranked[0]['expected_grade']} to {sensitivity_ranked[0]['next_grade']} "
            f"for +{sensitivity_ranked[0]['sgpa_gain_if_upgraded']:.4f} SGPA."
        ) if sensitivity_ranked else "All courses already at maximum grade.",
    }

    if student_id:
        log_calculation("/api/v1/course-plan", req.model_dump(), result, student_id)
    return result


@app.post("/api/v1/sensitivity", tags=["analysis"])
def sensitivity_analysis(req: SensitivityRequest,
                          student_id: Optional[int] = Query(default=None)):
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)
    target = D(req.target_cgpa)

    probes = sorted(set(req.probe_sgpa_values)) if req.probe_sgpa_values else [
        round(gs.passing_points + (gs.scale_max - gs.passing_points) / 9 * i, 2)
        for i in range(10)
    ]

    results = []
    for sgpa in probes:
        total = curr_credits + req.future_credits
        final_cgpa  = (curr_cgpa * curr_credits + D(sgpa) * req.future_credits) / total
        gap = target - final_cgpa
        recovery = ("none" if gap <= 0 else "minimal" if float(gap) < 0.1 else
                    "moderate" if float(gap) < 0.3 else "hard" if float(gap) < 0.6 else "very_hard")
        results.append({"avg_sgpa": round(sgpa, dp), "final_cgpa": round_dp(final_cgpa, dp),
                         "gap_to_target": round_dp(gap, dp+2),
                         "meets_target": final_cgpa >= target, "recovery_difficulty": recovery})

    total_c = curr_credits + req.future_credits
    marginal = round(req.future_credits / total_c, 4)
    result = {
        "marginal_sensitivity": marginal,
        "interpretation": f"Each 1.0 increase in average SGPA moves your CGPA by {marginal:.4f} ({marginal*100:.1f}%).",
        "sensitivity_table": results,
    }
    if student_id:
        log_calculation("/api/v1/sensitivity", req.model_dump(), result, student_id)
    return result


@app.post("/api/v1/simulate", tags=["analysis"])
def simulate(req: SimulateRequest,
             student_id: Optional[int] = Query(default=None)):
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)
    rw, rc = float(curr_cgpa) * curr_credits, curr_credits
    trajectory = []
    for s in req.planned_semesters:
        if not (0 <= s.sgpa <= gs.scale_max):
            raise HTTPException(422, f"Semester {s.semester_number}: sgpa {s.sgpa} out of [0, {gs.scale_max}]")
        rw += s.sgpa * s.credits
        rc += s.credits
        trajectory.append({
            "semester": s.semester_number, "sgpa": s.sgpa, "credits": s.credits,
            "projected_cgpa_after": round(rw / rc, dp),
            "nearest_grade": nearest_grade(s.sgpa, gs.grade_map),
        })
    result = {
        "starting_cgpa":        round_dp(curr_cgpa, dp),
        "final_projected_cgpa": trajectory[-1]["projected_cgpa_after"] if trajectory else round_dp(curr_cgpa, dp),
        "trajectory":           trajectory,
    }
    if student_id:
        log_calculation("/api/v1/simulate", req.model_dump(), result, student_id)
    return result


@app.post("/api/v1/compare-scenarios", tags=["analysis"])
def compare_scenarios(req: ScenarioCompareRequest,
                      student_id: Optional[int] = Query(default=None)):
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)
    future_credits = sum(req.credits_per_remaining_semester)
    scale_max = D(gs.scale_max)
    passing   = D(gs.passing_points)
    max_ach   = compute_max_achievable(curr_cgpa, curr_credits, future_credits, scale_max)
    rows = []
    for sc in req.scenarios:
        target   = D(sc.target_cgpa)
        if not (passing <= target <= scale_max):
            raise HTTPException(422, f"Target {sc.target_cgpa} outside [{gs.passing_points}, {gs.scale_max}]")
        req_sgpa = required_uniform_sgpa(curr_cgpa, curr_credits, target, future_credits)
        feas     = feasibility_report(req_sgpa, scale_max, passing, target, curr_cgpa, max_ach, dp)
        rows.append({
            "label": sc.label or f"Target {sc.target_cgpa}",
            "target_cgpa": sc.target_cgpa,
            "required_uniform_sgpa": round_dp(req_sgpa, dp+2),
            "feasible": feas["feasible"], "risk_level": feas["risk_level"],
            "already_achieved": feas.get("already_achieved", False),
            "message": feas["message"],
        })
    result = {
        "current_cgpa": round_dp(curr_cgpa, dp),
        "future_credits": future_credits,
        "max_achievable_cgpa": round_dp(max_ach, dp),
        "comparison_table": rows,
    }
    if student_id:
        log_calculation("/api/v1/compare-scenarios", req.model_dump(), result, student_id)
    return result


@app.post("/api/v1/academic-health", tags=["analysis"])
def academic_health(req: AcademicHealthRequest,
                    student_id: Optional[int] = Query(default=None)):
    gs      = req.grading_system
    dp      = gs.decimal_places
    history = req.academic_state.semester_history or []
    if not history:
        return {"academic_health_score": None, "consistency_score": None, "trend": "stable",
                "message": "No semester history provided."}
    sgpas = [s.sgpa for s in sorted(history, key=lambda s: s.semester_number)]
    std_dev     = statistics.stdev(sgpas) if len(sgpas) > 1 else 0.0
    consistency = max(0, round(100 * (1 - std_dev / gs.scale_max), 1))
    trend       = infer_trend(history)
    trend_bonus = {"improving": 5, "stable": 0, "declining": -10, "volatile": -5}[trend]
    mean_sgpa   = statistics.mean(sgpas)
    norm_mean   = (mean_sgpa - gs.passing_points) / (gs.scale_max - gs.passing_points) * 100
    health      = round(min(100, max(0, 0.6 * norm_mean + 0.4 * consistency + trend_bonus)), 1)
    last_3      = sgpas[-3:] if len(sgpas) >= 3 else sgpas
    stress      = ("high" if statistics.mean(last_3) < (gs.passing_points + (gs.scale_max - gs.passing_points) * 0.3)
                   else "moderate" if trend == "declining" else "low")
    result = {
        "academic_health_score": health, "consistency_score": consistency,
        "trend": trend, "stress_indicator": stress,
        "mean_sgpa": round(mean_sgpa, dp), "std_dev_sgpa": round(std_dev, dp+1),
        "semester_count": len(history),
        "interpretation": {
            "health":      "High" if health >= 70 else ("Moderate" if health >= 45 else "Needs attention"),
            "consistency": "High" if consistency >= 80 else ("Moderate" if consistency >= 55 else "Volatile"),
            "trend":       trend.capitalize(), "stress": stress.capitalize(),
        },
    }
    if student_id:
        log_calculation("/api/v1/academic-health", req.model_dump(), result, student_id)
    return result


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)
