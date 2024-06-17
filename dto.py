from pydantic import BaseModel

class InsuranceClaimDTO(BaseModel):
    KIDSDRIV: int
    AGE: float
    HOMEKIDS: int
    YOJ: float
    INCOME: float
    HOME_VAL: float
    TRAVTIME: int
    BLUEBOOK: float
    TIF: int
    OLDCLAIM: float
    CLM_FREQ: int
    MVR_PTS: int
    CLM_AMT: float
    CAR_AGE: float


# KIDSDRIV: 1 (One kid driving)
# AGE: 45 (Middle-aged individual)
# HOMEKIDS: 2 (Two kids at home)
# YOJ: 10 (10 years in current job)
# INCOME: 70000 (Income level)
# HOME_VAL: 300000 (Home value)
# TRAVTIME: 30 (Travel time to work)
# BLUEBOOK: 25000 (Estimated value of car)
# TIF: 5 (Years of the oldest vehicle)
# OLDCLAIM: 0 (No previous claims)
# CLM_FREQ: 2 (Previous claim frequency)
# MVR_PTS: 1 (Motor vehicle record points)
# CLM_AMT: 15000 (Claim amount)
# CAR_AGE: 4 (Age of car)