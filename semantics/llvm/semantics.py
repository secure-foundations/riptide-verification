from __future__ import annotations
from typing import Tuple, List, Dict
from dataclasses import dataclass

import semantics.smt as smt
from .ast import *


class StepResult:
    ...


@dataclass
class NextConfiguration(StepResult):
    config: Configuration


@dataclass
class FunctionReturn(StepResult):
    final_config: Configuration
    value: Optional[smt.SMTTerm]


@dataclass
class FunctionException(StepResult):
    reason: str


@dataclass
class MemoryUpdate:
    location: smt.SMTTerm
    value: smt.SMTTerm
    bit_width: int


@dataclass
class Configuration:
    module: Module
    function: Function

    current_block: str
    previous_block: Optional[str]
    current_instr_counter: int # Position of the current instruction within current_block

    variables: Dict[str, smt.SMTTerm]
    path_conditions: List[smt.SMTTerm]

    # For book keeping purposes
    memory_updates: List[MemoryUpdate] = field(default_factory=list)

    # memory_constraints will look like (m0 is the initial free memory variable)
    # m1 = store(m0, l0, v0)
    # m2 = store(m1, l1, v1)
    # ...
    memory_var: smt.SMTTerm = field(default_factory=lambda: Configuration.get_fresh_memory_var())
    memory_constraints: List[smt.SMTTerm] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []

        lines.append(f"current instruction: {self.current_block}:{self.current_instr_counter}")
        lines.append(f"previous block: {self.previous_block}")

        lines.append("variables:")
        for name, term in self.variables.items():
            lines.append(f"  {name}: {term}")

        lines.append("memory updates:")
        for update in self.memory_updates:
            lines.append(f"  {update.location} |-> {update.value} ({update.bit_width}-bit)")

        lines.append("path conditions:")
        for path_condition in self.path_conditions:
            lines.append(f"  {path_condition}")

        # max_line_width = max(map(len, lines))
        # lines = [ f"## {line}{' ' * (max_line_width - len(line))} ##" for line in lines ]
        # lines = [ "#" * (max_line_width + 6) ] + lines + [ "#" * (max_line_width + 6) ]

        return "\n".join(lines)
    
    @staticmethod
    def get_initial_configuration(module: Module, function: Function) -> Configuration:
        variables = {}

        for parameter in function.parameters.values():
            # TODO: right now this only supports pointers and integers
            variables[parameter.name] = smt.FreshSymbol(smt.BVType(parameter.get_type().get_bit_width()))

        return Configuration(
            module=module,
            function=function,
            current_block=tuple(function.blocks.keys())[0],
            previous_block=None,
            current_instr_counter=0,
            variables=variables,
            path_conditions=[],
        )
    
    @staticmethod
    def get_fresh_memory_var() -> smt.SMTTerm:
        return smt.FreshSymbol(smt.ArrayType(smt.BVType(WORD_WIDTH), smt.BVType(BYTE_WIDTH)))

    def copy(self) -> Configuration:
        return Configuration(
            self.module,
            self.function,
            self.current_block,
            self.previous_block,
            self.current_instr_counter,
            dict(self.variables),
            list(self.path_conditions),
            list(self.memory_updates),
            self.memory_var,
            list(self.memory_constraints),
        )

    def get_current_instruction(self) -> Instruction:
        return self.function.get_block(self.current_block).get_nth_instruction(self.current_instr_counter)

    def set_variable(self, name: str, value: smt.SMTTerm) -> None:
        self.variables[name] = value.simplify()

    def get_variable(self, name: str) -> smt.SMTTerm:
        assert name in self.variables, f"variable {name} not defined"
        return self.variables[name]

    def eval_value(self, value: Value) -> smt.SMTTerm:
        if isinstance(value, IntegerConstant):
            return smt.BVConst(value.value, value.type.bit_width)

        elif isinstance(value, Instruction):
            return self.get_variable(value.get_defined_variable())
        
        elif isinstance(value, FunctionParameter):
            return self.get_variable(value.name)

        else:
            assert False, f"evaluation of {value} not implemented"

    def check_feasibility(self) -> bool:
        with smt.Solver(name="z3") as solver:
            for path_condition in self.path_conditions:
                solver.add_assertion(path_condition)

            for memory_constraint in self.memory_constraints:
                solver.add_assertion(memory_constraint)

            # true for sat (i.e. feasible), false for unsat
            return solver.solve()

    def store_memory(self, location: smt.SMTTerm, value: smt.SMTTerm, bit_width: int) -> None:
        """
        Store value (of bit width bit_width) into the memory starting at `location`
        Location should be a BV of width WORD_WIDTH
        """

        self.memory_updates.append(MemoryUpdate(location, value, bit_width))

        # Align bit_width to BYTE_WIDTH
        aligned_bit_width = (bit_width + BYTE_WIDTH - 1) // BYTE_WIDTH * BYTE_WIDTH
        increase = aligned_bit_width - bit_width

        extended_value = smt.BVZExt(value, increase)

        new_memory_var = Configuration.get_fresh_memory_var()
        updated_memory_var = self.memory_var

        for i in range(aligned_bit_width // BYTE_WIDTH):
            # store the ith byte to dest + i
            updated_memory_var = smt.Store(
                updated_memory_var,
                smt.BVAdd(location, smt.BVConst(i, WORD_WIDTH)),
                smt.BVExtract(extended_value, i * BYTE_WIDTH, i * BYTE_WIDTH + BYTE_WIDTH - 1),
            )

        self.memory_constraints.append(smt.Equals(new_memory_var, updated_memory_var.simplify()))
        self.memory_var = new_memory_var
    
    def load_memory(self, location: smt.SMTTerm, bit_width: int) -> smt.SMTTerm:
        """
        Load a BV of width bit_width starting from the `location`
        """

        assert bit_width > 0

        aligned_bit_width = (bit_width + BYTE_WIDTH - 1) // BYTE_WIDTH * BYTE_WIDTH
        increase = aligned_bit_width - bit_width

        read_bytes = (
            smt.Select(self.memory_var, smt.BVAdd(location, smt.BVConst(i, WORD_WIDTH)))
            for i in range(aligned_bit_width // BYTE_WIDTH)
        )
        
        value = smt.BVConcat(*read_bytes)

        return smt.BVExtract(value, 0, bit_width - 1).simplify()

    def step(self) -> Tuple[StepResult, ...]:
        """
        Execute the next instruction

        Returns a non-empty tuple of symbolic branches

        The original configuration might be modified after the step function
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
    
        elif isinstance(instr, IntegerCompareInstruction):
            if instr.cond == "eq":
                result = smt.Equals(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )
            
            elif instr.cond == "sgt":
                result = smt.BVSGE(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "slt":
                result = smt.BVSLT(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            else:
                assert False, f"icmp condition {instr.cond} not implemented"

            self.set_variable(instr.name, smt.Ite(result, smt.BVConst(1, 1), smt.BVConst(0, 1)))
            self.current_instr_counter += 1
            return NextConfiguration(self),
    
        elif isinstance(instr, GetElementPointerInstruction):
            pointer = self.eval_value(instr.pointer)

            assert len(instr.indices) == 1, "unsupported"
            index = instr.indices[0]

            element_bit_width = instr.base_type.get_bit_width()
            aligned_element_byte_count = (element_bit_width + BYTE_WIDTH - 1) // BYTE_WIDTH

            index_value = self.eval_value(index)
            index_bit_width = index.get_type().get_bit_width()

            assert index_bit_width <= WORD_WIDTH
            if index_bit_width < WORD_WIDTH:
                index_value = smt.BVSExt(index_value, WORD_WIDTH - index_bit_width)

            pointer = smt.BVAdd(
                pointer,
                smt.BVMul(
                    smt.BVConst(aligned_element_byte_count, WORD_WIDTH),
                    index_value,
                ),
            )

            self.set_variable(instr.name, pointer)
            self.current_instr_counter += 1

            return NextConfiguration(self),
    
        elif isinstance(instr, LoadInstruction):
            bit_width = instr.base_type.get_bit_width()
            self.set_variable(
                instr.name,
                self.load_memory(self.eval_value(instr.pointer), bit_width),
            )
            self.current_instr_counter += 1
            return NextConfiguration(self),

        elif isinstance(instr, StoreInstruction):
            bit_width = instr.base_type.get_bit_width()
            self.store_memory(self.eval_value(instr.dest), self.eval_value(instr.value), bit_width)
            self.current_instr_counter += 1
            return NextConfiguration(self),
    
        elif isinstance(instr, SelectInstruction):
            cond = smt.Equals(self.eval_value(instr.cond), smt.BVConst(0, 1))

            self.current_instr_counter += 1
            other = self.copy()
            self.path_conditions.append(smt.Not(cond).simplify())
            other.path_conditions.append(cond.simplify())

            self.set_variable(instr.name, self.eval_value(instr.left))
            other.set_variable(instr.name, self.eval_value(instr.right))

            if not self.check_feasibility():
                # other.path_conditions implies cond
                other.path_conditions.pop()
                return NextConfiguration(other),

            if not other.check_feasibility():
                self.path_conditions.pop()
                return NextConfiguration(self),
            
            return NextConfiguration(self), NextConfiguration(other)

        elif isinstance(instr, PhiInstruction):
            assert self.previous_block in instr.branches, \
                   f"block label {self.current_block} not in the list of phi branches"
            self.set_variable(instr.name, self.eval_value(instr.branches[self.previous_block].value))
            self.current_instr_counter += 1
            return NextConfiguration(self),
    
        elif isinstance(instr, JumpInstruction):
            self.previous_block = self.current_block
            self.current_block = instr.label
            self.current_instr_counter = 0
            return NextConfiguration(self),

        elif isinstance(instr, BranchInstruction):
            cond = smt.Equals(self.eval_value(instr.cond), smt.BVConst(0, 1))

            # Br is always at the end of a block
            self.current_instr_counter = 0

            self.previous_block = self.current_block
            other = self.copy()

            self.current_block = instr.true_label
            other.current_block = instr.false_label
            
            self.path_conditions.append(smt.Not(cond).simplify())
            other.path_conditions.append(cond.simplify())

            if not self.check_feasibility():
                # other.path_conditions implies cond
                other.path_conditions.pop()
                return NextConfiguration(other),

            if not other.check_feasibility():
                self.path_conditions.pop()
                return NextConfiguration(self),
            
            return NextConfiguration(self), NextConfiguration(other)

        elif isinstance(instr, ReturnInstruction):
            if instr.value is not None:
                return_value = self.eval_value(instr.value)
            else:
                return_value = None

            return FunctionReturn(self, return_value),

        else:
            assert False, f"{instr.get_full_string()} not implemented"
