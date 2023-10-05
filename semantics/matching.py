from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field
from collections import OrderedDict

from . import smt


class MatchingResult:
    def merge(self, other: MatchingResult) -> MatchingResult:
        raise NotImplementedError()
    
    @staticmethod
    def match_smt_terms(pattern: smt.SMTTerm, value: smt.SMTTerm) -> MatchingResult:
        if pattern.is_symbol():
            return MatchingSuccess(OrderedDict({ pattern: value }))
        else:
            assert pattern.get_free_variables().issubset(value.get_free_variables())
            return MatchingSuccess(OrderedDict(), smt.Equals(
                pattern,
                value,
            ))


@dataclass
class MatchingSuccess:
    substitution: OrderedDict[smt.SMTTerm, smt.SMTTerm] = field(default_factory=OrderedDict)
    condition: smt.SMTTerm = field(default_factory=smt.TRUE)

    def merge(self, other: MatchingResult) -> MatchingResult:
        if isinstance(other, MatchingFailure):
            return MatchingFailure(f"failed to merge with other matching failure: {other.reason}")
        
        assert isinstance(other, MatchingSuccess)
        assert set(self.substitution.keys()).isdisjoint(set(other.substitution.keys())), \
               f"overlapping keys {set(self.substitution.keys())}, {set(other.substitution.keys())}"
        return MatchingSuccess(
            OrderedDict(tuple(self.substitution.items()) + tuple(other.substitution.items())),
            smt.And(self.condition, other.condition),
        )


@dataclass
class MatchingFailure:
    reason: Optional[str] = None

    def merge(self, other: MatchingResult) -> MatchingResult:
        if isinstance(other, MatchingFailure):
            return MatchingFailure(f"failed to merge two matching failures: {self.reason}; {other.reason}")
        else:
            return MatchingFailure(f"failed to merge with a matching failure: {self.reason}")
