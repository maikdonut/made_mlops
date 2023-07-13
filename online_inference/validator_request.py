from typing import Literal

from pydantic import BaseModel, validator
from fastapi import HTTPException


class Patient(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @staticmethod
    def check_valid_value(name, l_border, r_border, value):
        if value < l_border or value > r_border:
            raise HTTPException(
                status_code=400,
                detail=f'ERROR: {name} must be in range ({r_border} - {l_border})'
            )
        return value

    @validator('age')
    def valid_age(cls, value):
        return cls.check_valid_value('age', 0, 100, value)

    @validator('trestbps')
    def check_trestbps(cls, value):
        return cls.check_valid_value('trestbps', 90, 200, value)

    @validator('chol')
    def check_chol(cls, value):
        return cls.check_valid_value('chol', 125, 565, value)

    @validator('thalach')
    def check_thalach(cls, value):
        return cls.check_valid_value('thalach', 70, 205, value)

    @validator('oldpeak')
    def check_oldpeak(cls, value):
        return cls.check_valid_value('oldpeak', 0, 6.2, value)
