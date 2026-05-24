from sqlalchemy import Column, Integer, Float, String, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Student(Base):
    __tablename__ = "students"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    semesters  = relationship("SemesterRecord", back_populates="student")
    plans      = relationship("PlanResult", back_populates="student")

class SemesterRecord(Base):
    __tablename__ = "semester_records"
    id               = Column(Integer, primary_key=True)
    student_id       = Column(Integer, ForeignKey("students.id"))
    semester_number  = Column(Integer)
    sgpa             = Column(Float)
    credits          = Column(Integer)
    student          = relationship("Student", back_populates="semesters")

class PlanResult(Base):
    __tablename__ = "plan_results"
    id             = Column(Integer, primary_key=True)
    student_id     = Column(Integer, ForeignKey("students.id"), nullable=True)
    request_data   = Column(JSON)   # stores the full request payload
    result_data    = Column(JSON)   # stores the full API response
    created_at     = Column(DateTime, default=datetime.utcnow)
    student        = relationship("Student", back_populates="plans")