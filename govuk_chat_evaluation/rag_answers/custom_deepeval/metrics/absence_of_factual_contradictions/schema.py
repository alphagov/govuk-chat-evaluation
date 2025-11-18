from typing import Literal
from pydantic import BaseModel


class Verdict(BaseModel):
    verdict: Literal["yes", "no", "idk"]
    reason: str | None = None


class VerdictCollection(BaseModel):
    verdicts: list[Verdict]

    def score_verdicts(self) -> float:
        if len(self.verdicts) == 0:
            return 1.0

        verdicts_count = sum(1 for verdict in self.verdicts if verdict.verdict != "no")
        return float(verdicts_count / len(self.verdicts))

    def contradiction_reasons(self) -> list[str]:
        return [
            verdict.reason
            for verdict in self.verdicts
            if verdict.verdict == "no" and verdict.reason
        ]


class TruthCollection(BaseModel):
    truths: list[str]


class ClaimCollection(BaseModel):
    claims: list[str]


class ScoreReason(BaseModel):
    reason: str
