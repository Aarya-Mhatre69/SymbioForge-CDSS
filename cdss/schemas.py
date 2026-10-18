"""Input validation (FR1/FR2). Field-level errors come from pydantic; bad input never reaches the models."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Sex = Literal["male", "female"]
CP = Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
RestECG = Literal["normal", "lv hypertrophy", "st-t abnormality"]
Slope = Literal["upsloping", "flat", "downsloping"]
Thal = Literal["normal", "fixed defect", "reversable defect"]

TRAIN_AGE_RANGE = (28, 77)


class PatientInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Sex
    cp: CP
    trestbps: int = Field(ge=80, le=220, description="resting BP, mmHg")
    chol: int = Field(ge=100, le=600, description="serum cholesterol, mg/dl")
    fbs: bool = Field(description="fasting blood sugar > 120 mg/dl")
    restecg: RestECG
    thalch: int = Field(ge=60, le=220, description="max heart rate, bpm")
    exang: bool
    oldpeak: float = Field(ge=-3.0, le=7.0)
    slope: Optional[Slope] = None
    ca: Optional[int] = Field(default=None, ge=0, le=3)
    thal: Optional[Thal] = None

    @field_validator("slope", "ca", "thal", mode="before")
    @classmethod
    def blank_is_missing(cls, v):
        return None if v in ("", "?", "unknown") else v

    def missing(self) -> list[str]:
        return [f for f in ("slope", "ca", "thal") if getattr(self, f) is None]


def field_errors(exc) -> dict[str, str]:
    return {".".join(map(str, e["loc"])): e["msg"] for e in exc.errors()}
