from pydantic import BaseModel, Field


class CoherenceJudgement(BaseModel):
    score: int = Field(..., description="Rubric score from 1 (worst) to 5 (best).")
    reason: str = Field(
        ..., description="Focused explanation referencing structure and logical flow."
    )
