from dataclasses import dataclass
from datetime import datetime

@dataclass
class Student:
    id: int = None
    student_id: str = ""
    full_name: str = ""

@dataclass
class AttendanceRecord:
    id: int = None
    student_id: str = ""
    timestamp: datetime = None
    confidence: float = 0.0