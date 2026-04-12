from pydantic import BaseModel
from typing import List, Optional

class Patient(BaseModel):
    id: int
    symptoms: str
    severity: int
    assigned: bool = False

class Observation(BaseModel):
    patients: List[dict]
    available_beds: int
    step_count: int

class Action(BaseModel):
    patient_id: int
    assign_bed: bool

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[dict] = None