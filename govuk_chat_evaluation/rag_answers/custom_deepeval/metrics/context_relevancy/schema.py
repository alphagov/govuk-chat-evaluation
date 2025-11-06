from typing import List, Literal
from pydantic import BaseModel, Field


class Verdict(BaseModel):
    verdict: Literal["yes", "idk", "no"]
    reason: str | None = Field(default=None)


class VerdictCollection(BaseModel):
    verdicts: List[Verdict]

    def score_verdicts(self) -> float:
        if len(self.verdicts) == 0:
            return 1.0

        quality_count = sum(1 for verdict in self.verdicts if verdict.verdict != "no")
        return float(quality_count / len(self.verdicts))

    def unmet_needs(self) -> List[str]:
        unmet_needs = []

        for verdict in self.verdicts:
            if verdict.verdict == "no" and verdict.reason:
                unmet_needs.append(verdict.reason)

        return unmet_needs


class Truth(BaseModel):
    context: str
    facts: List[str]


class TruthCollection(BaseModel):
    truths: List[Truth]


class InformationNeedsCollection(BaseModel):
    information_needs: List[str]


class ScoreReason(BaseModel):
    reason: str
