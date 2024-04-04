from __future__ import annotations
from typing import Tuple, List, Dict, Callable
from dataclasses import dataclass

import semantics.smt as smt

from semantics.matching import *
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

    def __str__(self) -> str:
        return f"{self.location} |-> {self.value} ({self.bit_width}-bit)"


@dataclass
class Configuration:
    module: Module
    function: Function

    current_block: str
    previous_block: Optional[str]
    current_instr_counter: int # Position of the current instruction within current_block

    variables: OrderedDict[str, smt.SMTTerm]
    path_conditions: List[smt.SMTTerm]

    # For book keeping purposes
    memory_updates: List[MemoryUpdate] = field(default_factory=list)
    memory: smt.SMTTerm = field(default_factory=lambda: Configuration.get_fresh_memory_var())

    solver: Optional[smt.Solver] = None

    uncommitted_phi_assignments: List[Tuple[str, smt.SMTTerm]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = []

        lines.append(f"current instruction: {self.current_block}:{self.current_instr_counter}")
        lines.append(f"previous block: {self.previous_block}")

        lines.append("variables:")
        for name, term in self.variables.items():
            lines.append(f"  {name}: {term}")

        lines.append("memory updates:")
        for update in self.memory_updates:
            lines.append(f"  {update}")

        lines.append("path conditions:")
        for path_condition in self.path_conditions:
            lines.append(f"  {path_condition}")

        # max_line_width = max(map(len, lines))
        # lines = [ f"## {line}{' ' * (max_line_width - len(line))} ##" for line in lines ]
        # lines = [ "#" * (max_line_width + 6) ] + lines + [ "#" * (max_line_width + 6) ]

        lines = [ "|| " + line for line in lines ]
        lines = ["===== llvm state begin ====="] + lines + ["===== llvm state end ====="]

        return "\n".join(lines)

    def match(self, other: Configuration) -> MatchingResult:
        """
        Use self as a pattern and try to find an assignment to variables
        in self so that self = other

        This assumes some constraints on self:
        - memory_updates should be empty
        - all SMT terms in variables are SMT variables or have
          free variables contained in the other configuration

        Returns (if successful) the matching substitution
        and a condition for the matching to be valid ("other.path_condition => self.path_condition")
        all variables in this condition is implicitly universally quantified
        (so to check it, we need to check the unsat of the negation)
        """

        assert self.module == other.module and self.function == other.function
        assert len(self.memory_updates) == 0
        assert len(self.uncommitted_phi_assignments) == 0

        if self.current_block != other.current_block:
            return MatchingFailure("unmatched current block")

        if self.previous_block != other.previous_block:
            return MatchingFailure("unmatched previous block")

        if self.current_instr_counter != other.current_instr_counter:
            return MatchingFailure("unmatched instruction counter")

        assert len(other.uncommitted_phi_assignments) == 0

        result = MatchingSuccess()

        for var_name, term in self.variables.items():
            if var_name not in other.variables:
                return MatchingFailure(f"variable {var_name} is not defined")

            result = result.merge(MatchingResult.match_smt_terms(term, other.variables[var_name]))
            if isinstance(result, MatchingFailure):
                return result

        assert isinstance(result, MatchingSuccess)

        assert self.memory.is_symbol()
        result.substitution[self.memory] = other.memory

        # TODO: this is ignoring memory constraints
        substituted_path_conditions = (
            path_condition.substitute(result.substitution)
            for path_condition in self.path_conditions
        )

        return MatchingSuccess(result.substitution, smt.Implies(
            smt.And(*other.path_conditions),
            smt.And(result.condition, *substituted_path_conditions),
        ))

    @staticmethod
    def get_initial_configuration(module: Module, function: Function, solver: Optional[smt.Solevr] = None) -> Configuration:
        variables = OrderedDict()

        # TODO: right now this only supports pointers and integers

        for parameter in function.parameters.values():
            variables[parameter.name] = smt.FreshSymbol(smt.BVType(parameter.get_type().get_bit_width()), "llvm_param_" + parameter.name.replace(".", "_").strip("%") + "_%d")

        for var, instr in function.definitions.items():
            variables[var] = smt.FreshSymbol(smt.BVType(instr.get_type().get_bit_width()), "llvm_var_" + var.replace(".", "_").strip("%") + "_%d")

        return Configuration(
            module=module,
            function=function,
            current_block=tuple(function.blocks.keys())[0],
            previous_block=None,
            current_instr_counter=0,
            variables=variables,
            path_conditions=[],
            solver=solver,
        )

    @staticmethod
    def get_fresh_memory_var() -> smt.SMTTerm:
        # return smt.FreshSymbol(smt.ArrayType(smt.BVType(WORD_WIDTH), smt.BVType(BYTE_WIDTH)))
        return smt.FreshSymbol(smt.ArrayType(smt.BVType(WORD_WIDTH), smt.BVType(WORD_WIDTH)), "llvm_mem_%d")

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
            self.memory,
            self.solver,
            list(self.uncommitted_phi_assignments),
        )

    def get_current_instruction(self) -> Instruction:
        return self.function.get_block(self.current_block).get_nth_instruction(self.current_instr_counter)

    def set_variable(self, name: str, value: smt.SMTTerm) -> None:
        self.variables[name] = value.simplify()

    def get_variable(self, name: str) -> smt.SMTTerm:
        assert name in self.variables, f"variable {name} not defined"
        return self.variables[name]

    def has_variable(self, name: str) -> bool:
        return name in self.variables

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
        return smt.check_sat(self.path_conditions, self.solver)

    def store_memory(self, location: smt.SMTTerm, value: smt.SMTTerm, bit_width: int) -> None:
        """
        Store value (of bit width bit_width) into the memory starting at `location`
        Location should be a BV of width WORD_WIDTH
        """

        assert bit_width == WORD_WIDTH, f"unsupported memory write of {bit_width} bit"

        self.memory_updates.append(MemoryUpdate(location, value, bit_width))
        self.memory = smt.Store(self.memory, location, value)

        # # Align bit_width to BYTE_WIDTH
        # aligned_bit_width = (bit_width + BYTE_WIDTH - 1) // BYTE_WIDTH * BYTE_WIDTH
        # increase = aligned_bit_width - bit_width

        # extended_value = smt.BVZExt(value, increase)

        # new_memory_var = Configuration.get_fresh_memory_var()
        # updated_memory_var = self.memory_var

        # for i in range(aligned_bit_width // BYTE_WIDTH):
        #     # store the ith byte to dest + i
        #     updated_memory_var = smt.Store(
        #         updated_memory_var,
        #         smt.BVAdd(location, smt.BVConst(i, WORD_WIDTH)),
        #         smt.BVExtract(extended_value, i * BYTE_WIDTH, i * BYTE_WIDTH + BYTE_WIDTH - 1),
        #     )

    def load_memory(self, location: smt.SMTTerm, bit_width: int) -> smt.SMTTerm:
        """
        Load a BV of width bit_width starting from the `location`
        """

        assert bit_width == WORD_WIDTH, f"unsupported memory read of bit width {bit_width}"
        assert bit_width > 0

        # aligned_bit_width = (bit_width + BYTE_WIDTH - 1) // BYTE_WIDTH * BYTE_WIDTH
        # increase = aligned_bit_width - bit_width

        # read_bytes = (
        #     smt.Select(self.memory_var, smt.BVAdd(location, smt.BVConst(i, WORD_WIDTH)))
        #     for i in range(aligned_bit_width // BYTE_WIDTH)
        # )

        # value = smt.BVConcat(*read_bytes)

        # return smt.BVExtract(value, 0, bit_width - 1).simplify()

        return smt.Select(self.memory, location)

    def step(self) -> Tuple[StepResult, ...]:
        """
        Execute the next instruction

        Returns a non-empty tuple of symbolic branches

        The original configuration might be modified after the step function
        so do not use it after step is called.
        """

        instr = self.get_current_instruction()

        binary_op_semantics: Dict[str, Callable[[smt.SMTTerm, smt.SMTTerm], smt.SMTTerm]] = {
            AddInstruction: smt.BVAdd,
            SubInstruction: smt.BVSub,
            MulInstruction: smt.BVMul,
            AndInstruction: smt.BVAnd,
            OrInstruction: smt.BVOr,
            XorInstruction: smt.BVXor,
            ShlInstruction: smt.BVLShl,
            LshrInstruction: smt.BVLShr,
            AshrInstruction: smt.BVAShr,
        }

        # First non-phi instruction, commit all phi assignments
        if not isinstance(instr, PhiInstruction) and len(self.uncommitted_phi_assignments) != 0:
            for var, value in self.uncommitted_phi_assignments:
                self.set_variable(var, value)
            self.uncommitted_phi_assignments = []

        if instr.__class__ in binary_op_semantics:
            func = binary_op_semantics[instr.__class__]
            self.set_variable(instr.name, func(
                self.eval_value(instr.left),
                self.eval_value(instr.right),
            ))
            self.current_instr_counter += 1
            return NextConfiguration(self),

        elif isinstance(instr, ZextInstruction):
            assert instr.to_type.bit_width >= instr.from_type.bit_width
            self.set_variable(instr.name, smt.BVZExt(
                self.eval_value(instr.value),
                instr.to_type.bit_width - instr.from_type.bit_width,
            ))
            self.current_instr_counter += 1
            return NextConfiguration(self),

        elif isinstance(instr, SextInstruction):
            assert instr.to_type.bit_width >= instr.from_type.bit_width
            self.set_variable(instr.name, smt.BVSExt(
                self.eval_value(instr.value),
                instr.to_type.bit_width - instr.from_type.bit_width,
            ))
            self.current_instr_counter += 1
            return NextConfiguration(self),

        elif isinstance(instr, IntegerCompareInstruction):
            if instr.cond == "eq":
                result = smt.Equals(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "sge":
                result = smt.BVSGE(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "sgt":
                result = smt.BVSGT(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "sle":
                result = smt.BVSLE(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "slt":
                result = smt.BVSLT(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "uge":
                result = smt.BVUGE(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "ugt":
                result = smt.BVUGT(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "ule":
                result = smt.BVULE(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            elif instr.cond == "ult":
                result = smt.BVULT(
                    self.eval_value(instr.left),
                    self.eval_value(instr.right),
                )

            else:
                assert False, f"icmp condition {instr.cond} not implemented"

            # More eager branching
            # self.current_instr_counter += 1
            # other = self.copy()

            # self.set_variable(instr.name, smt.BVConst(1, 1))
            # other.set_variable(instr.name, smt.BVConst(0, 1))

            # self.path_conditions.append(result.simplify())
            # other.path_conditions.append(smt.Not(result).simplify())

            # if not self.check_feasibility():
            #     # other.path_conditions implies cond
            #     other.path_conditions.pop()
            #     return NextConfiguration(other),

            # if not other.check_feasibility():
            #     self.path_conditions.pop()
            #     return NextConfiguration(self),

            # return NextConfiguration(self), NextConfiguration(other)

            self.set_variable(instr.name, smt.Ite(result, smt.BVConst(1, 1), smt.BVConst(0, 1)))
            self.current_instr_counter += 1
            return NextConfiguration(self),

        elif isinstance(instr, GetElementPointerInstruction):
            pointer = self.eval_value(instr.pointer)

            assert len(instr.indices) == 1, "unsupported"
            index = instr.indices[0]

            element_bit_width = instr.base_type.get_bit_width()
            index_bit_width = index.get_type().get_bit_width()

            assert element_bit_width == index_bit_width == WORD_WIDTH, "unsupported"

            # aligned_element_byte_count = (element_bit_width + BYTE_WIDTH - 1) // BYTE_WIDTH

            # index_value = self.eval_value(index)
            # index_bit_width = index.get_type().get_bit_width()

            # assert index_bit_width <= WORD_WIDTH
            # if index_bit_width < WORD_WIDTH:
            #     index_value = smt.BVSExt(index_value, WORD_WIDTH - index_bit_width)

            # pointer = smt.BVAdd(
            #     pointer,
            #     smt.BVMul(
            #         smt.BVConst(aligned_element_byte_count, WORD_WIDTH),
            #         index_value,
            #     ),
            # )

            self.set_variable(instr.name, smt.BVAdd(pointer, self.eval_value(index)))
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

        elif isinstance(instr, CallInstruction):
            # TODO: these implementations are a bit hacky
            if instr.function_name == "@cgra_load32":
                assert len(instr.arguments) >= 1
                self.set_variable(
                    instr.name,
                    self.load_memory(self.eval_value(instr.arguments[0][1]), 32),
                )
                self.current_instr_counter += 1
                return NextConfiguration(self),

            elif instr.function_name == "@cgra_store32":
                assert len(instr.arguments) >= 2
                self.store_memory(self.eval_value(instr.arguments[1][1]), self.eval_value(instr.arguments[0][1]), 32)
                self.set_variable(instr.name, smt.BVConst(1, 32))
                self.current_instr_counter += 1
                return NextConfiguration(self),

            elif instr.function_name == "@llvm.fshl.i32":
                assert len(instr.arguments) == 3
                a = self.eval_value(instr.arguments[0][1])
                b = self.eval_value(instr.arguments[1][1])
                c = self.eval_value(instr.arguments[2][1])
                self.set_variable(instr.name, smt.BVExtract(smt.BVLShl(smt.BVConcat(a, b), smt.BVZExt(c, WORD_WIDTH)), WORD_WIDTH, 2 * WORD_WIDTH - 1))
                self.current_instr_counter += 1
                return NextConfiguration(self),

            else:
                assert False, f"function {instr.function_name} not implemented"

        elif isinstance(instr, PhiInstruction):
            assert self.previous_block in instr.branches, f"block label {self.previous_block} not in the list of phi branches"

            # All phi instructions should be run "in parallel"
            # e.g.
            # a = phi c
            # b = phi a // this should take the value of a at the loop header, instead of the new value
            self.uncommitted_phi_assignments.append((instr.name, self.eval_value(instr.branches[self.previous_block].value)))
            # self.set_variable(instr.name, self.eval_value(instr.branches[self.previous_block].value))
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
