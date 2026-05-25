"""
CGPA Tracker API  —  v2
=======================
A genuine academic planning engine, not just a calculator.

Endpoints (all under /api/v1/):
  GET  /grading-presets
  POST /validate-grading-system
  POST /calculate-plan          ← optimization engine (LP + fallback)
  POST /course-plan             ← course-level SGPA planning
  POST /sensitivity             ← delta_cgpa / delta_sgpa analysis
  POST /simulate                ← what-if trajectory
  POST /compare-scenarios       ← side-by-side target comparison
  POST /academic-health         ← health + consistency metrics

Design principles:
  - Decimal arithmetic throughout (no float drift)
  - LP optimization via scipy (fallback to uniform if unavailable)
  - Full explainability: every number justified
  - Versioned under /api/v1/
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal, ROUND_HALF_UP, getcontext
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

getcontext().prec = 28  # high-precision Decimal arithmetic

# ── Try to import scipy; fall back gracefully ──────────────────────────────
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
    version="2.0.0",
    description="Academic planning engine: optimization, sensitivity analysis, course-level planning.",
    openapi_tags=[
        {"name": "setup",    "description": "Grading system configuration"},
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


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MODELS
# ══════════════════════════════════════════════════════════════════════════════

class GradeEntry(BaseModel):
    grade:       str
    points:      float
    min_percent: Optional[float] = None

    model_config = {"json_schema_extra": {"example": {"grade": "A+", "points": 9.0, "min_percent": 80}}}


class GradingSystem(BaseModel):
    scale_max:       float  = Field(..., gt=0, description="Highest possible grade point")
    passing_points:  float  = Field(..., ge=0, description="Minimum grade points to pass")
    grade_map:       list[GradeEntry]
    credit_range:    list[int] = Field(default=[15, 25], description="[min, max] credits per semester")
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
            raise ValueError(
                f"Highest grade point in grade_map must equal scale_max ({self.scale_max})"
            )
        if len(self.credit_range) != 2 or self.credit_range[0] <= 0 or self.credit_range[1] < self.credit_range[0]:
            raise ValueError("credit_range must be [min, max] with min > 0")
        return self


class SemesterRecord(BaseModel):
    semester_number: int
    sgpa:            float
    credits:         int = Field(..., gt=0)

    model_config = {"json_schema_extra": {"example": {"semester_number": 1, "sgpa": 7.8, "credits": 22}}}


class AcademicState(BaseModel):
    """
    Two input modes (mutually exclusive):
      A) semester_history  — full list of past semesters
      B) current_cgpa + total_credits_earned  — summary mode

    Leave both empty for a freshman with no history.
    """
    semester_history:      Optional[list[SemesterRecord]] = None
    current_cgpa:          Optional[float]                = None
    total_credits_earned:  Optional[int]                  = Field(default=None, gt=0)

    @model_validator(mode="after")
    def exactly_one_mode(self):
        has_history = bool(self.semester_history)
        has_summary = self.current_cgpa is not None and self.total_credits_earned is not None
        if has_history and has_summary:
            raise ValueError(
                "Supply either semester_history OR (current_cgpa + total_credits_earned), not both."
            )
        return self


# ══════════════════════════════════════════════════════════════════════════════
# DECIMAL MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def D(x) -> Decimal:
    """Safely convert to Decimal."""
    return Decimal(str(x))


def weighted_cgpa(records: list[dict]) -> Decimal:
    """Weighted average: sum(sgpa*credits) / sum(credits). Returns 0 if no records."""
    tw = sum(D(r["sgpa"]) * D(r["credits"]) for r in records)
    tc = sum(D(r["credits"]) for r in records)
    return tw / tc if tc else D("0")


def extract_earned(state: AcademicState) -> tuple[Decimal, int]:
    """Returns (current_cgpa as Decimal, total_credits_earned as int)."""
    if state.semester_history:
        records = [{"sgpa": s.sgpa, "credits": s.credits} for s in state.semester_history]
        return weighted_cgpa(records), sum(r["credits"] for r in records)
    if state.current_cgpa is not None:
        return D(state.current_cgpa), state.total_credits_earned
    return D("0"), 0


def required_uniform_sgpa(
    curr_cgpa: Decimal,
    curr_credits: int,
    target: Decimal,
    future_credits: int,
) -> Decimal:
    if future_credits == 0:
        return D("0")
    total = curr_credits + future_credits
    return (target * total - curr_cgpa * curr_credits) / future_credits


def round_dp(val: Decimal, dp: int) -> float:
    """Round Decimal to dp places, return float."""
    quantize_str = D("0." + "0" * dp) if dp > 0 else D("1")
    return float(val.quantize(quantize_str, rounding=ROUND_HALF_UP))


def nearest_grade(sgpa: float, grade_map: list[GradeEntry]) -> str:
    sorted_grades = sorted(grade_map, key=lambda g: g.points, reverse=True)
    for g in sorted_grades:
        if sgpa >= g.points:
            return g.grade
    return sorted_grades[-1].grade


def infer_trend(history: list[SemesterRecord]) -> str:
    """
    Detect performance trend from semester history.
    Returns: 'improving' | 'declining' | 'stable' | 'volatile'
    """
    if not history or len(history) < 2:
        return "stable"
    scores = [s.sgpa for s in sorted(history, key=lambda s: s.semester_number)]
    diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
    avg_diff = statistics.mean(diffs)
    variance = statistics.variance(scores) if len(scores) > 1 else 0

    if variance > 1.5:
        return "volatile"
    if avg_diff > 0.15:
        return "improving"
    if avg_diff < -0.15:
        return "declining"
    return "stable"


# ══════════════════════════════════════════════════════════════════════════════
# FEASIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_max_achievable(
    curr_cgpa: Decimal, curr_credits: int,
    future_credits: int, scale_max: Decimal,
) -> Decimal:
    total = curr_credits + future_credits
    return (curr_cgpa * curr_credits + scale_max * future_credits) / total


def feasibility_report(
    req: Decimal,
    scale_max: Decimal,
    passing: Decimal,
    target: Decimal,
    curr_cgpa: Decimal,
    max_achievable: Decimal,
    dp: int,
) -> dict:
    if curr_cgpa >= target:
        return {
            "feasible": True,
            "already_achieved": True,
            "risk_level": "none",
            "message": "You have already met or exceeded your target CGPA.",
        }
    if req > scale_max:
        return {
            "feasible": False,
            "already_achieved": False,
            "risk_level": "impossible",
            "message": (
                f"Mathematically impossible. Even scoring {scale_max} every semester "
                f"yields at most {round_dp(max_achievable, dp)}. "
                f"Consider adjusting your target or extending your timeline."
            ),
            "max_achievable_cgpa": round_dp(max_achievable, dp),
        }

    gap = float(req - passing)
    scale_span = float(scale_max - passing)
    difficulty_ratio = gap / scale_span if scale_span > 0 else 0

    if difficulty_ratio > 0.85:
        risk = "extreme"
        msg = f"Requires near-perfect performance ({round_dp(req, dp+2)}) every semester. Very high burnout risk."
    elif difficulty_ratio > 0.65:
        risk = "high"
        msg = f"Demanding but achievable with consistent high effort ({round_dp(req, dp+2)} SGPA)."
    elif difficulty_ratio > 0.35:
        risk = "moderate"
        msg = f"Realistic goal. Requires steady effort ({round_dp(req, dp+2)} SGPA per semester)."
    else:
        risk = "low"
        msg = f"Comfortable target. Even {round_dp(passing, dp+2)} SGPA in some semesters leaves buffer."

    return {
        "feasible": True,
        "already_achieved": False,
        "risk_level": risk,
        "required_uniform_sgpa": round_dp(req, dp + 2),
        "message": msg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class EffortPattern(str, Enum):
    uniform       = "uniform"       # minimize variance
    front_loaded  = "front_loaded"  # minimize effort later
    back_loaded   = "back_loaded"   # ease in, push later
    stepped_up    = "stepped_up"    # linear ramp up
    stepped_down  = "stepped_down"  # linear ramp down
    custom        = "custom"        # user-constrained


class SemesterConstraint(BaseModel):
    semester_index: int         = Field(..., ge=0, description="0-based index into remaining semesters")
    max_sgpa:       Optional[float] = None
    min_sgpa:       Optional[float] = None
    label:          Optional[str]   = None   # e.g. "internship semester"


def optimize_plan_scipy(
    curr_cgpa: float,
    curr_credits: int,
    target: float,
    credits_per_sem: list[int],
    scale_max: float,
    passing: float,
    pattern: EffortPattern,
    constraints: list[SemesterConstraint],
    desired_buffer: float = 0.0,
) -> tuple[list[float], str]:
    """
    Uses scipy.optimize to find optimal SGPA allocation.
    pattern=uniform     → minimize sum of squared deviations from mean (QP via minimize)
    pattern=front_loaded → minimize sum of s_i * (n-i) weights (penalize high later semesters)
    pattern=back_loaded  → minimize sum of s_i * i weights (penalize high early)
    pattern=stepped_up   → minimize sum of (s_i - s_{i-1})^2 where s_i >= s_{i-1}
    pattern=stepped_down → same but s_i <= s_{i-1}

    Returns (sgpa_list, method_used)
    """
    n = len(credits_per_sem)
    future_credits = sum(credits_per_sem)

    # Build per-semester bounds
    lower = [max(passing, c.min_sgpa) if c and c.min_sgpa else passing for c in _align_constraints(n, constraints)]
    upper = [min(scale_max, c.max_sgpa) if c and c.max_sgpa else scale_max for c in _align_constraints(n, constraints)]

    # CGPA constraint: sum(s_i * c_i) / (curr + future) >= target + buffer
    effective_target = target + desired_buffer
    rhs = effective_target * (curr_credits + future_credits) - curr_cgpa * curr_credits

    # All objectives are quadratic → use scipy.optimize.minimize with SLSQP
    if pattern == EffortPattern.uniform:
        def obj(s):
            mean = sum(s) / n
            return sum((x - mean)**2 for x in s)
    elif pattern == EffortPattern.front_loaded:
        def obj(s):
            return sum(s[i] * (n - i) for i in range(n))
    elif pattern == EffortPattern.back_loaded:
        def obj(s):
            return sum(s[i] * (i + 1) for i in range(n))
    elif pattern == EffortPattern.stepped_up:
        def obj(s):
            # minimize variance of differences (smooth ramp), penalize decreases
            diffs = [s[i+1] - s[i] for i in range(n-1)]
            return sum((d - abs(d))**2 * 100 for d in diffs) + statistics.variance(diffs) if len(diffs) > 1 else 0
    elif pattern == EffortPattern.stepped_down:
        def obj(s):
            diffs = [s[i] - s[i+1] for i in range(n-1)]
            return sum((d - abs(d))**2 * 100 for d in diffs) + (statistics.variance(diffs) if len(diffs) > 1 else 0)
    else:
        def obj(s):
            mean = sum(s) / n
            return sum((x - mean)**2 for x in s)

    constraints_scipy = [
        {"type": "ineq", "fun": lambda s: sum(s[i] * credits_per_sem[i] for i in range(n)) - rhs}
    ]
    bounds = [(lower[i], upper[i]) for i in range(n)]
    x0 = [min(upper[i], max(lower[i], rhs / future_credits)) for i in range(n)]

    result = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints_scipy,
                      options={"ftol": 1e-10, "maxiter": 1000})

    if result.success:
        return [max(lower[i], min(upper[i], result.x[i])) for i in range(n)], "scipy_slsqp"
    # fallback: uniform distribution
    return _fallback_uniform(curr_cgpa, curr_credits, target, credits_per_sem, scale_max, passing, constraints), "fallback_uniform"


def _align_constraints(n: int, constraints: list[SemesterConstraint]) -> list[Optional[SemesterConstraint]]:
    aligned = [None] * n
    for c in constraints:
        if 0 <= c.semester_index < n:
            aligned[c.semester_index] = c
    return aligned


def _fallback_uniform(
    curr_cgpa: float,
    curr_credits: int,
    target: float,
    credits: list[int],
    scale_max: float,
    passing: float,
    constraints: list[SemesterConstraint],
) -> list[float]:
    req = (target * (curr_credits + sum(credits)) - curr_cgpa * curr_credits) / sum(credits)
    req = max(passing, min(scale_max, req))
    aligned = _align_constraints(len(credits), constraints)
    result = []
    for i, c in enumerate(aligned):
        lo = c.min_sgpa if c and c.min_sgpa else passing
        hi = c.max_sgpa if c and c.max_sgpa else scale_max
        result.append(max(lo, min(hi, req)))
    return result


def build_plan_from_sgpas(
    sgpas: list[float],
    credits_per_sem: list[int],
    curr_cgpa: Decimal,
    curr_credits: int,
    dp: int,
    grade_map: list[GradeEntry],
    sem_constraints: list[SemesterConstraint],
) -> list[dict]:
    """Convert a list of SGPA floats into a rich per-semester plan."""
    aligned_constraints = _align_constraints(len(sgpas), sem_constraints)
    running_weighted = float(curr_cgpa) * curr_credits
    running_credits  = curr_credits
    plan = []

    for i, (sgpa, credits) in enumerate(zip(sgpas, credits_per_sem)):
        running_weighted += sgpa * credits
        running_credits  += credits
        cgpa_after = running_weighted / running_credits
        c = aligned_constraints[i]

        plan.append({
            "semester": i + 1,
            "target_sgpa": round(sgpa, dp + 2),
            "credits": credits,
            "projected_cgpa_after": round(cgpa_after, dp),
            "nearest_grade": nearest_grade(sgpa, grade_map),
            "constraint_label": c.label if c else None,
            "is_constrained": c is not None,
        })

    return plan


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def quality_points_breakdown(
    curr_cgpa: Decimal,
    curr_credits: int,
    target: Decimal,
    future_credits: int,
) -> dict:
    total_credits = curr_credits + future_credits
    earned_qp     = curr_cgpa * curr_credits
    required_total_qp = target * total_credits
    remaining_qp_needed = required_total_qp - earned_qp

    return {
        "current_quality_points":          round(float(earned_qp), 2),
        "required_total_quality_points":   round(float(required_total_qp), 2),
        "remaining_quality_points_needed": round(float(remaining_qp_needed), 2),
        "future_credits":                  future_credits,
        "total_credits_at_graduation":     total_credits,
        "quality_points_per_future_credit_needed": round(float(remaining_qp_needed / future_credits), 4) if future_credits else 0,
        "explanation": (
            f"You have earned {round(float(earned_qp), 2)} quality points over {curr_credits} credits. "
            f"To reach CGPA {float(target):.2f} by graduation ({total_credits} total credits), "
            f"you need {round(float(required_total_qp), 2)} total quality points — "
            f"meaning you must earn {round(float(remaining_qp_needed), 2)} more quality points "
            f"across your remaining {future_credits} credits "
            f"({round(float(remaining_qp_needed / future_credits), 4) if future_credits else 0} per credit)."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(
    risk_level: str,
    trend: str,
    pattern: EffortPattern,
    remaining_sems: int,
    req_uniform: float,
    scale_max: float,
    passing: float,
) -> list[str]:
    recs = []
    span = scale_max - passing

    if risk_level == "impossible":
        recs.append("Your current target is out of reach. Consider extending your program by one semester or adjusting the target CGPA.")
        return recs

    if risk_level in ("extreme", "high"):
        recs.append(f"Required SGPA of {req_uniform:.2f} leaves almost no room for error. Plan every semester carefully.")
        if remaining_sems <= 2:
            recs.append("With few semesters left, every single course matters. Prioritize high-credit courses for maximum impact.")
        else:
            recs.append("Use the front-loaded strategy: perform at your peak now while you have more semesters as buffer.")

    if trend == "declining":
        recs.append("Your SGPA has been declining. Investigate the cause — increasing course load, elective difficulty, or external factors — before committing to an aggressive target.")
    elif trend == "improving":
        recs.append("Your SGPA trend is positive. Stay consistent — your trajectory already supports your goal.")
    elif trend == "volatile":
        recs.append("Volatile performance history increases risk. A stepped-up strategy may help you build consistency before pushing hard.")

    if pattern == EffortPattern.back_loaded:
        recs.append("Back-loading is psychologically risky: if life events disrupt your final semesters, recovery becomes impossible.")

    if (req_uniform - passing) / span < 0.2 if span > 0 else False:
        recs.append("Your target is conservative. Consider raising it slightly — you likely have more capacity than you think.")

    if remaining_sems >= 4:
        recs.append("You have ample time. Small consistent improvements each semester are more sustainable than last-minute surges.")

    return recs or ["Your plan looks balanced. Stay consistent and revisit this calculator each semester."]


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS FOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    grading_system:               GradingSystem
    academic_state:               AcademicState
    target_cgpa:                  float = Field(..., description="Desired final CGPA")
    remaining_semesters:          int   = Field(..., gt=0)
    credits_per_remaining_semester: list[int]
    effort_pattern:               EffortPattern = EffortPattern.uniform
    semester_constraints:         list[SemesterConstraint] = []
    desired_buffer:               float = Field(default=0.0, ge=0.0, description="Extra CGPA cushion above target")

    @model_validator(mode="after")
    def validate_credits_and_target(self):
        if len(self.credits_per_remaining_semester) != self.remaining_semesters:
            raise ValueError(
                f"credits_per_remaining_semester must have {self.remaining_semesters} entries, "
                f"got {len(self.credits_per_remaining_semester)}"
            )
        if any(c <= 0 for c in self.credits_per_remaining_semester):
            raise ValueError("All semester credit values must be positive")
        gs = self.grading_system
        if not (gs.passing_points <= self.target_cgpa <= gs.scale_max):
            raise ValueError(
                f"target_cgpa ({self.target_cgpa}) must be in [{gs.passing_points}, {gs.scale_max}]"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "grading_system": {
                    "scale_max": 10.0, "passing_points": 4.0, "decimal_places": 2,
                    "credit_range": [15, 30],
                    "grade_map": [
                        {"grade": "O", "points": 10.0}, {"grade": "A+", "points": 9.0},
                        {"grade": "A", "points": 8.0},  {"grade": "B+", "points": 7.0},
                        {"grade": "B", "points": 6.0},  {"grade": "C",  "points": 5.0},
                        {"grade": "P", "points": 4.0},  {"grade": "F",  "points": 0.0},
                    ],
                },
                "academic_state": {"current_cgpa": 7.5, "total_credits_earned": 120},
                "target_cgpa": 8.5,
                "remaining_semesters": 4,
                "credits_per_remaining_semester": [22, 22, 20, 20],
                "effort_pattern": "uniform",
                "semester_constraints": [
                    {"semester_index": 1, "max_sgpa": 8.0, "label": "internship semester"}
                ],
                "desired_buffer": 0.1,
            }
        }
    }


class CourseEntry(BaseModel):
    name:           str
    credits:        int   = Field(..., gt=0)
    expected_grade: str   = Field(..., description="Grade label, e.g. 'A+', 'B'")


class CoursePlanRequest(BaseModel):
    grading_system:  GradingSystem
    academic_state:  AcademicState
    target_sgpa:     float
    courses:         list[CourseEntry] = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "grading_system": {
                    "scale_max": 10.0, "passing_points": 4.0, "decimal_places": 2,
                    "credit_range": [15, 30],
                    "grade_map": [
                        {"grade": "O", "points": 10.0}, {"grade": "A+", "points": 9.0},
                        {"grade": "A", "points": 8.0},  {"grade": "B+", "points": 7.0},
                        {"grade": "B", "points": 6.0},  {"grade": "F",  "points": 0.0},
                    ],
                },
                "academic_state": {"current_cgpa": 7.5, "total_credits_earned": 120},
                "target_sgpa": 8.5,
                "courses": [
                    {"name": "Data Structures", "credits": 4, "expected_grade": "A+"},
                    {"name": "Operating Systems", "credits": 4, "expected_grade": "B+"},
                    {"name": "Maths III", "credits": 3, "expected_grade": "A"},
                ],
            }
        }
    }


class SensitivityRequest(BaseModel):
    grading_system:  GradingSystem
    academic_state:  AcademicState
    target_cgpa:     float
    future_credits:  int = Field(..., gt=0)
    probe_sgpa_values: Optional[list[float]] = None   # if None, auto-generate grid


class SimulateRequest(BaseModel):
    grading_system:   GradingSystem
    academic_state:   AcademicState
    planned_semesters: list[SemesterRecord]


class ScenarioItem(BaseModel):
    target_cgpa: float
    label:       Optional[str] = None


class ScenarioCompareRequest(BaseModel):
    grading_system:              GradingSystem
    academic_state:              AcademicState
    scenarios:                   list[ScenarioItem] = Field(..., min_length=2, max_length=8)
    remaining_semesters:         int = Field(..., gt=0)
    credits_per_remaining_semester: list[int]

    @model_validator(mode="after")
    def credits_match(self):
        if len(self.credits_per_remaining_semester) != self.remaining_semesters:
            raise ValueError("credits_per_remaining_semester length must equal remaining_semesters")
        return self


class AcademicHealthRequest(BaseModel):
    grading_system: GradingSystem
    academic_state: AcademicState   # must have semester_history for full metrics


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/grading-presets", tags=["setup"])
def grading_presets():
    return {
        "presets": [
            {
                "name": "10-Point Scale (India — most universities)",
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
                    {"grade": "A",  "points": 4.0, "min_percent": 93},
                    {"grade": "A-", "points": 3.7, "min_percent": 90},
                    {"grade": "B+", "points": 3.3, "min_percent": 87},
                    {"grade": "B",  "points": 3.0, "min_percent": 83},
                    {"grade": "B-", "points": 2.7, "min_percent": 80},
                    {"grade": "C+", "points": 2.3, "min_percent": 77},
                    {"grade": "C",  "points": 2.0, "min_percent": 73},
                    {"grade": "D",  "points": 1.0, "min_percent": 60},
                    {"grade": "F",  "points": 0.0, "min_percent": 0},
                ],
            },
            {
                "name": "Percentage-Based (100-Point)",
                "scale_max": 100.0, "passing_points": 40.0,
                "decimal_places": 1, "credit_range": [15, 30],
                "grade_map": [
                    {"grade": "O", "points": 90.0},
                    {"grade": "A", "points": 75.0},
                    {"grade": "B", "points": 60.0},
                    {"grade": "C", "points": 50.0},
                    {"grade": "P", "points": 40.0},
                    {"grade": "F", "points": 0.0},
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
        "credit_range": gs.credit_range,
        "decimal_places": gs.decimal_places,
    }


@app.post("/api/v1/calculate-plan", tags=["planning"])
def calculate_plan(req: PlanRequest):
    """
    Core endpoint. Runs the optimization engine (scipy SLSQP when available,
    analytic fallback otherwise) to produce the optimal per-semester SGPA plan.
    Returns: feasibility, optimized plan, quality-points breakdown, recommendations.
    """
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)

    # Validate current CGPA against scale
    if curr_credits > 0 and not (D("0") <= curr_cgpa <= D(gs.scale_max)):
        raise HTTPException(422, f"current_cgpa ({float(curr_cgpa):.2f}) outside [0, {gs.scale_max}]")

    future_credits = sum(req.credits_per_remaining_semester)
    target         = D(req.target_cgpa)
    scale_max      = D(gs.scale_max)
    passing        = D(gs.passing_points)

    req_uniform = required_uniform_sgpa(curr_cgpa, curr_credits, target, future_credits)
    max_ach     = compute_max_achievable(curr_cgpa, curr_credits, future_credits, scale_max)

    # Trend from history (if available)
    trend = infer_trend(req.academic_state.semester_history or [])

    # Feasibility
    feas = feasibility_report(req_uniform, scale_max, passing, target, curr_cgpa, max_ach, dp)

    # Quality points explainability
    qp = quality_points_breakdown(curr_cgpa, curr_credits, target, future_credits)

    # Optimization
    if SCIPY_AVAILABLE and feas["feasible"] and not feas.get("already_achieved"):
        sgpas, method = optimize_plan_scipy(
            float(curr_cgpa), curr_credits, float(target + D(req.desired_buffer)),
            req.credits_per_remaining_semester,
            float(scale_max), float(passing),
            req.effort_pattern, req.semester_constraints, req.desired_buffer,
        )
    elif feas.get("already_achieved"):
        sgpas = [float(curr_cgpa)] * req.remaining_semesters
        method = "already_achieved"
    else:
        sgpas = _fallback_uniform(
            float(curr_cgpa), curr_credits, float(target),
            req.credits_per_remaining_semester,
            float(scale_max), float(passing),
            req.semester_constraints,
        )
        method = "fallback_uniform"

    plan = build_plan_from_sgpas(
        sgpas, req.credits_per_remaining_semester,
        curr_cgpa, curr_credits, dp, gs.grade_map, req.semester_constraints,
    )

    # Trajectory for charts
    trajectory = [
        {
            "semester": row["semester"],
            "projected_cgpa": row["projected_cgpa_after"],
            "target_sgpa": row["target_sgpa"],
        }
        for row in plan
    ]

    # Recommendations
    recs = generate_recommendations(
        feas["risk_level"],
        trend,
        req.effort_pattern,
        req.remaining_semesters,
        float(req_uniform),
        float(scale_max),
        float(passing),
    )

    return {
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
        "trajectory":               trajectory,
        "recommendations":          recs,
    }


@app.post("/api/v1/course-plan", tags=["planning"])
def course_plan(req: CoursePlanRequest):
    """
    Course-level planning: given a list of courses with expected grades,
    compute the semester's SGPA and compare it against your target.
    Also shows sensitivity: which course improvements move the needle most.
    """
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)

    # Build grade → points map
    grade_to_points: dict[str, float] = {g.grade: g.points for g in gs.grade_map}
    sorted_grades = sorted(gs.grade_map, key=lambda g: g.points, reverse=True)

    # Resolve each course
    course_details = []
    total_credits = 0
    total_weighted = D("0")

    for c in req.courses:
        if c.expected_grade not in grade_to_points:
            raise HTTPException(
                422,
                f"Grade '{c.expected_grade}' for course '{c.name}' not found in grade_map. "
                f"Valid grades: {list(grade_to_points.keys())}",
            )
        pts = grade_to_points[c.expected_grade]
        total_weighted += D(pts) * D(c.credits)
        total_credits  += c.credits

        # Find next higher grade
        next_grade = None
        for g in sorted_grades:
            if g.points > pts:
                next_grade = g
                break

        # Impact of improving this course by one grade
        impact_one_grade = None
        if next_grade:
            impact_one_grade = round(
                (next_grade.points - pts) * c.credits / total_credits, dp + 2
            )

        course_details.append({
            "name":                c.name,
            "credits":             c.credits,
            "expected_grade":      c.expected_grade,
            "grade_points":        pts,
            "weighted_contribution": round(float(D(pts) * D(c.credits) / D(total_credits)), dp + 2),
            "next_grade":          next_grade.grade if next_grade else None,
            "sgpa_gain_if_upgraded": impact_one_grade,
        })

    projected_sgpa = float(total_weighted / D(total_credits))

    # Post-semester CGPA
    new_cgpa = (float(curr_cgpa) * curr_credits + projected_sgpa * total_credits) / (curr_credits + total_credits)

    # Gap vs target SGPA
    gap = req.target_sgpa - projected_sgpa

    # Sort courses by sensitivity (biggest gain per credit upgrade)
    sensitivity_ranked = sorted(
        [c for c in course_details if c["sgpa_gain_if_upgraded"] is not None],
        key=lambda c: c["sgpa_gain_if_upgraded"],
        reverse=True,
    )

    return {
        "semester_summary": {
            "total_credits":     total_credits,
            "projected_sgpa":    round(projected_sgpa, dp),
            "target_sgpa":       req.target_sgpa,
            "sgpa_gap":          round(gap, dp + 2),
            "meets_target":      projected_sgpa >= req.target_sgpa,
            "new_cgpa_if_achieved": round(new_cgpa, dp),
        },
        "courses":            course_details,
        "sensitivity_ranking": sensitivity_ranked,
        "advice": (
            f"Upgrade '{sensitivity_ranked[0]['name']}' from "
            f"{sensitivity_ranked[0]['expected_grade']} to {sensitivity_ranked[0]['next_grade']} "
            f"for the highest SGPA gain (+{sensitivity_ranked[0]['sgpa_gain_if_upgraded']:.4f})."
        ) if sensitivity_ranked else "All courses are already at maximum grade.",
    }


@app.post("/api/v1/sensitivity", tags=["analysis"])
def sensitivity_analysis(req: SensitivityRequest):
    """
    Computes how each possible SGPA value across remaining semesters affects final CGPA.
    Returns a grid of (sgpa → final_cgpa) plus recovery difficulty scores.
    """
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)
    target = D(req.target_cgpa)

    # Auto-generate probe points if not provided
    if req.probe_sgpa_values:
        probes = sorted(set(req.probe_sgpa_values))
    else:
        step = (gs.scale_max - gs.passing_points) / 9
        probes = [round(gs.passing_points + step * i, 2) for i in range(10)]

    results = []
    for sgpa in probes:
        sgpa_d = D(sgpa)
        total_credits = curr_credits + req.future_credits
        final_cgpa = (curr_cgpa * curr_credits + sgpa_d * req.future_credits) / total_credits
        gap_to_target = target - final_cgpa

        if gap_to_target <= 0:
            recovery = "none"
        elif float(gap_to_target) < 0.1:
            recovery = "minimal"
        elif float(gap_to_target) < 0.3:
            recovery = "moderate"
        elif float(gap_to_target) < 0.6:
            recovery = "hard"
        else:
            recovery = "very_hard"

        results.append({
            "avg_sgpa":            round(sgpa, dp),
            "final_cgpa":          round_dp(final_cgpa, dp),
            "gap_to_target":       round_dp(gap_to_target, dp + 2),
            "meets_target":        final_cgpa >= target,
            "recovery_difficulty": recovery,
        })

    # Marginal sensitivity: how much does 1 unit of SGPA move the final CGPA?
    # d(CGPA) / d(SGPA) = future_credits / total_credits
    total = curr_credits + req.future_credits
    marginal_sensitivity = round(req.future_credits / total, 4)

    return {
        "marginal_sensitivity": marginal_sensitivity,
        "interpretation": (
            f"Each 1.0 increase in average SGPA across remaining semesters "
            f"moves your final CGPA by {marginal_sensitivity:.4f} ({marginal_sensitivity*100:.1f}%)."
        ),
        "sensitivity_table": results,
    }


@app.post("/api/v1/simulate", tags=["analysis"])
def simulate(req: SimulateRequest):
    """What-if: user enters their own SGPA plan, gets projected CGPA trajectory."""
    gs  = req.grading_system
    dp  = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)

    running_weighted = float(curr_cgpa) * curr_credits
    running_credits  = curr_credits
    trajectory = []

    for s in req.planned_semesters:
        if not (0 <= s.sgpa <= gs.scale_max):
            raise HTTPException(422, f"Semester {s.semester_number}: sgpa {s.sgpa} out of [0, {gs.scale_max}]")
        running_weighted += s.sgpa * s.credits
        running_credits  += s.credits
        cgpa = running_weighted / running_credits
        trajectory.append({
            "semester":              s.semester_number,
            "sgpa":                  s.sgpa,
            "credits":               s.credits,
            "projected_cgpa_after":  round(cgpa, dp),
            "nearest_grade":         nearest_grade(s.sgpa, gs.grade_map),
        })

    return {
        "starting_cgpa":        round_dp(curr_cgpa, dp),
        "final_projected_cgpa": trajectory[-1]["projected_cgpa_after"] if trajectory else round_dp(curr_cgpa, dp),
        "trajectory":           trajectory,
    }


@app.post("/api/v1/compare-scenarios", tags=["analysis"])
def compare_scenarios(req: ScenarioCompareRequest):
    """
    Side-by-side comparison of multiple target CGPAs.
    Shows required uniform SGPA, feasibility, and risk level for each.
    """
    gs = req.grading_system
    dp = gs.decimal_places
    curr_cgpa, curr_credits = extract_earned(req.academic_state)
    future_credits = sum(req.credits_per_remaining_semester)
    scale_max = D(gs.scale_max)
    passing   = D(gs.passing_points)
    max_ach   = compute_max_achievable(curr_cgpa, curr_credits, future_credits, scale_max)

    rows = []
    for sc in req.scenarios:
        target = D(sc.target_cgpa)
        if not (passing <= target <= scale_max):
            raise HTTPException(422, f"Scenario target {sc.target_cgpa} outside [{gs.passing_points}, {gs.scale_max}]")
        req_sgpa = required_uniform_sgpa(curr_cgpa, curr_credits, target, future_credits)
        feas     = feasibility_report(req_sgpa, scale_max, passing, target, curr_cgpa, max_ach, dp)

        rows.append({
            "label":                  sc.label or f"Target {sc.target_cgpa}",
            "target_cgpa":            sc.target_cgpa,
            "required_uniform_sgpa":  round_dp(req_sgpa, dp + 2),
            "feasible":               feas["feasible"],
            "risk_level":             feas["risk_level"],
            "already_achieved":       feas.get("already_achieved", False),
            "message":                feas["message"],
        })

    return {
        "current_cgpa":      round_dp(curr_cgpa, dp),
        "future_credits":    future_credits,
        "max_achievable_cgpa": round_dp(max_ach, dp),
        "comparison_table":  rows,
    }


@app.post("/api/v1/academic-health", tags=["analysis"])
def academic_health(req: AcademicHealthRequest):
    """
    Compute academic health score, consistency score, trend, and stress indicators
    from a student's semester history.
    Requires semester_history in academic_state for full metrics.
    """
    gs = req.grading_system
    dp = gs.decimal_places
    history = req.academic_state.semester_history or []

    if not history:
        return {
            "academic_health_score":  None,
            "consistency_score":      None,
            "trend":                  "stable",
            "message":                "No semester history provided. Health metrics require at least 2 semesters of data.",
        }

    sgpas = [s.sgpa for s in sorted(history, key=lambda s: s.semester_number)]

    # Consistency: 100 = perfectly consistent, 0 = max volatility
    std_dev = statistics.stdev(sgpas) if len(sgpas) > 1 else 0.0
    consistency = max(0, round(100 * (1 - std_dev / gs.scale_max), 1))

    # Trend
    trend = infer_trend(history)

    # Trajectory direction multiplier
    trend_bonus = {"improving": 5, "stable": 0, "declining": -10, "volatile": -5}[trend]

    # Academic health: weighted by mean SGPA, consistency, trend
    mean_sgpa = statistics.mean(sgpas)
    normalized_mean = (mean_sgpa - gs.passing_points) / (gs.scale_max - gs.passing_points) * 100
    health = round(min(100, max(0, 0.6 * normalized_mean + 0.4 * consistency + trend_bonus)), 1)

    # Stress indicator
    last_3 = sgpas[-3:] if len(sgpas) >= 3 else sgpas
    if statistics.mean(last_3) < (gs.passing_points + (gs.scale_max - gs.passing_points) * 0.3):
        stress = "high"
    elif trend == "declining":
        stress = "moderate"
    else:
        stress = "low"

    return {
        "academic_health_score": health,
        "consistency_score":     consistency,
        "trend":                 trend,
        "stress_indicator":      stress,
        "mean_sgpa":             round(mean_sgpa, dp),
        "std_dev_sgpa":          round(std_dev, dp + 1),
        "semester_count":        len(history),
        "interpretation": {
            "health":      "High" if health >= 70 else ("Moderate" if health >= 45 else "Needs attention"),
            "consistency": "High" if consistency >= 80 else ("Moderate" if consistency >= 55 else "Volatile"),
            "trend":       trend.capitalize(),
            "stress":      stress.capitalize(),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)
