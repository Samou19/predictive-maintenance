# Schéma des données (schemas.py)
from pydantic import BaseModel

class CycleInput(BaseModel):
    ps2_mean: float
    ps2_std: float
    ps2_min: float
    ps2_max: float
    fs1_mean: float
    fs1_std: float
    fs1_min: float
    fs1_max: float