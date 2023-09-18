from __future__ import annotations
from typing import Tuple, List, Dict
from dataclasses import dataclass

import smt
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
    path_conditions: List[smt.SMTTerm]

    def copy(self) -> Configuration:
        return Configuration(
            self.module,
            self.function,
            self.current_block,
            self.previous_block,
            self.current_instr_counter,
            dict(self.variables),
            list(self.path_conditions),
        )

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

    def check_feasibility(self) -> bool:
        with smt.Solver(name="z3") as solver:
            for path_condition in self.path_conditions:
                solver.add_assertion(path_condition)

            # true for sat (i.e. feasible), false for unsat
            return solver.solve()

    def step(self) -> Tuple[StepResult, ...]:
        """
        Execute the next instruction

        Returns a non-empty tuple of symbolic branches

        The original configuration MIGHT be modified after the step function
        so do not use it after step is called.
        """

        instr = self.get_current_instruction()

        if isinstance(instr, AddInstruction):
            self.set_variable(instr.name, smt.BVAdd(
                self.eval_value(instr.left),
                self.eval_value(instr.right),
            ))
            self.current_instr_counter += 1
            return NextConfiguration(self),

        elif isinstance(instr, MulInstruction):
            self.set_variable(instr.name, smt.BVMul(
                self.eval_value(instr.left),
                self.eval_value(instr.right),
            ))
            self.current_instr_counter += 1
            return NextConfiguration(self),
    
        elif isinstance(instr, PhiInstruction):
            assert self.current_block in instr.branches, \
                   f"block label {self.current_block} not in the list of phi branches"
            self.set_variable(instr.name, self.eval_value(instr.branches[self.current_block].value))
            self.current_instr_counter += 1
            return NextConfiguration(self),
    
        elif isinstance(instr, BranchInstruction):
            # Would always be at the end of a block
            cond = self.eval_value(instr.cond)
            self.current_instr_counter = 0

            self.previous_block = self.current_block
            other = self.copy()

            self.current_block = instr.true_label
            other.current_block = instr.false_label
            
            self.path_conditions.append(cond)
            other.path_conditions.append(cond)

            if not self.check_feasibility():
                return NextConfiguration(other)

            if not other.check_feasibility():
                return NextConfiguration(self)
            
            return NextConfiguration(self), NextConfiguration(other)

        else:
            assert False, f"{instr.get_full_string()} not implemented"
