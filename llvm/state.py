from __future__ import annotations

import smt

from typing import Dict
from dataclasses import dataclass


@dataclass
class Configuration:
    current_block: str
    previous_block: str
    current_instr_counter: int

    registers: Dict[str, smt.SMTTerm]
    path_condition: smt.SMTTerm
