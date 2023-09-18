from __future__ import annotations
from typing import Tuple

import smt

from typing import Dict
from dataclasses import dataclass

from .ast import *


class StepResult:
    ...


@dataclass
class NextConfiguration(StepResult):
    config: Configuration


@dataclass
class FunctionReturn(StepResult):
    final_config: Configuration
    value: smt.SMTTerm


@dataclass
class FunctionException(StepResult):
    reason: str


@dataclass
class Configuration:
    module: Module
    function: Function

    current_block: str
    previous_block: str
    current_instr_counter: int # Position of the current instruction within current_block

    variables: Dict[str, smt.SMTTerm]
    path_condition: smt.SMTTerm

    def get_current_instruction(self) -> Instruction:
        return self.function.get_block(self.current_block).get_nth_instruction(self.current_instr_counter)

    def set_variable(self, name: str, value: smt.SMTTerm) -> None:
        self.variables[name] = value

    def get_variable(self, name: str) -> smt.SMTTerm:
        assert name in self.variables, f"variable {name} not defined"
        return self.variables[name]

    def eval_value(self, value: Value) -> smt.SMTTerm:
        if isinstance(value, IntegerConstant):
            return smt.BV(value.value % (2 ** value.type.bit_width), value.type.bit_width)

        elif isinstance(value, Instruction):
            return self.get_variable(value.get_defined_variable())
        
        else:
            assert False, f"evaluation of {value} not implemented"

    def step(self) -> Tuple[Tuple[smt.SMTTerm, StepResult], ...]:
        """
        Execute the next instruction
        """

        instr = self.get_current_instruction()

        if isinstance(instr, AddInstruction):
            self.set_variable(instr.name, smt.BVAdd(
                self.eval_value(instr.left),
                self.eval_value(instr.right),
            ))

        else:
            assert False, f"{instr.get_full_string()} not implemented"
