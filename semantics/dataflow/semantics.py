from __future__ import annotations

from typing import Tuple, Optional, List, Generator, Type, Dict, Mapping, Any, Set, Callable, Iterable
from dataclasses import dataclass, field
from collections import OrderedDict

import inspect

import semantics.smt as smt

from semantics.matching import *
from . import permission
from .graph import DataflowGraph, ProcessingElement, ConstantValue, FunctionArgument


WORD_WIDTH = 32

PERMISSION_PREFIX_EXEC = "exec-"
PERMISSION_PREFIX_INTERNAL = "internal-"
PERMISSION_PREFIX_HOLD_CONSTANT = "hold-"
PERMISSION_PREFIX_CONSTANT = "const-"
PERMISSION_PREFIX_OUTPUT_CHANNEL = "output-"


TransitionFunction = Callable[..., Any]


@dataclass
class ChannelId:
    id: int


@dataclass
class Branching:
    condition: smt.SMTTerm
    true_branch: TransitionFunction
    false_branch: TransitionFunction


class Operator:
    OPERATOR_IMPL_MAP: Dict[str, Type[Operator]] = {}

    @staticmethod
    def implement(name: str):
        def wrapper(cls: Type[Operator]):
            Operator.OPERATOR_IMPL_MAP[name] = cls
            return cls
        return wrapper

    def __init__(self, pe: ProcessingElement, permission: permission.Variable, transition: Optional[TransitionFunction] = None):
        self.pe = pe
        self.internal_permission: permission.Variable = permission
        self.current_transition: TransitionFunction = transition or type(self).start

    def is_at_start(self) -> bool:
        return self.current_transition == type(self).start

    def transition_to(self, transition: TransitionFunction):
        self.current_transition = transition

    def start(self, config: Configuration):
        raise NotImplementedError()

    def copy(self) -> Operator:
        return type(self)(self.pe, self.internal_permission, self.current_transition)

    def match(self, other: Operator) -> MatchingResult:
        assert self.pe == other.pe, "invalid pattern"
        # TODO: permission matching
        if self.current_transition == other.current_transition:
            return MatchingSuccess()
        else:
            return MatchingFailure(f"unmatched transition at operator {self.pe.id}")


@Operator.implement("ARITH_CFG_OP_ID")
class IdOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0)) -> ChannelId(0):
        return a


@Operator.implement("ARITH_CFG_OP_EQ")
class EqOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.Equals(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_ADD")
class AddOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVAdd(a, b)


@Operator.implement("ARITH_CFG_OP_SUB")
class AddOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVSub(a, b)


@Operator.implement("ARITH_CFG_OP_AND")
class AndOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVAnd(a, b)


@Operator.implement("ARITH_CFG_OP_OR")
class OrOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVOr(a, b)


@Operator.implement("ARITH_CFG_OP_XOR")
class XorOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVXor(a, b)


@Operator.implement("ARITH_CFG_OP_SHL")
class ShlOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVLShl(a, b)


@Operator.implement("ARITH_CFG_OP_LSHR")
class LshrOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVLShr(a, b)


@Operator.implement("ARITH_CFG_OP_ASHR")
class AshrOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVAShr(a, b)


@Operator.implement("ARITH_CFG_OP_FSHL")
class FshlOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1), c: ChannelId(2)) -> ChannelId(0):
        return smt.BVExtract(smt.BVLShl(smt.BVConcat(a, b), smt.BVZExt(c, WORD_WIDTH)), WORD_WIDTH, 2 * WORD_WIDTH - 1)


@Operator.implement("ARITH_CFG_OP_GEP")
class GEPOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.BVAdd(a, b)


@Operator.implement("MUL_CFG_OP_MUL")
class MulOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        # print("mul", a, b)
        return smt.BVMul(a, b)


@Operator.implement("ARITH_CFG_OP_SGE")
class SignedGreaterThanOrEqualOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVSGE(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_SGT")
class SignedGreaterThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVSGT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_SLE")
class SignedLessThanOrEqualOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVSLE(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_SLT")
class SignedLessThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVSLT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_UGE")
class UnsignedGreaterThanOrEqualOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVUGE(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_UGT")
class UnsignedGreaterThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVUGT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_ULE")
class UnsignedLessThanOrEqualOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVULE(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_ULT")
class UnsignedLessThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVULT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("CF_CFG_OP_SELECT")
class SelectOperator(Operator):
    def start(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), SelectOperator.false, SelectOperator.true)

    def true(self, config: Configuration, a: ChannelId(1), b: ChannelId(2)) -> ChannelId(0):
        self.transition_to(SelectOperator.start)
        return a

    def false(self, config: Configuration, a: ChannelId(1), b: ChannelId(2)) -> ChannelId(0):
        self.transition_to(SelectOperator.start)
        return b


@Operator.implement("CF_CFG_OP_MERGE")
class MergeOperator(Operator):
    def start(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), MergeOperator.false, MergeOperator.true)

    def true(self, config: Configuration, a: ChannelId(1)) -> ChannelId(0):
        self.transition_to(MergeOperator.start)
        return a

    def false(self, config: Configuration, b: ChannelId(2)) -> ChannelId(0):
        self.transition_to(MergeOperator.start)
        return b


@Operator.implement("CF_CFG_OP_CARRY")
class CarryOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(1)) -> ChannelId(0):
        self.transition_to(CarryOperator.loop)
        return a

    def loop(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), CarryOperator.pass_b, CarryOperator.start)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), CarryOperator.start, CarryOperator.pass_b)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def pass_b(self, config: Configuration, b: ChannelId(2)) -> ChannelId(0):
        self.transition_to(CarryOperator.loop)
        return b


@Operator.implement("CF_CFG_OP_STEER")
class SteerOperator(Operator):
    def start(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), SteerOperator.pass_value, SteerOperator.discard_value)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), SteerOperator.discard_value, SteerOperator.pass_value)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def pass_value(self, config: Configuration, value: ChannelId(1)) -> ChannelId(0):
        self.transition_to(SteerOperator.start)
        return value

    def discard_value(self, config: Configuration, value: ChannelId(1)):
        self.transition_to(SteerOperator.start)


@Operator.implement("CF_CFG_OP_INVARIANT")
class InvariantOperator(Operator):
    def __init__(self, *args):
        super().__init__(*args)
        self.value: Optional[smt.SMTTerm] = None

    def copy(self) -> InvariantOperator:
        copied = super().copy()
        copied.value = self.value
        return copied

    def match(self, other: Operator) -> MatchingResult:
        result = super().match(other)
        assert isinstance(other, InvariantOperator)
        if self.value is None and other.value is None:
            return result
        elif (self.value is None) != (other.value is None):
            return result.merge(MatchingFailure(f"invariant value not matched at operator {self.pe.id}"))
        else:
            assert self.value is not None and other.value is not None
            return result.merge(MatchingResult.match_smt_terms(self.value, other.value))

    def start(self, config: Configuration, value: ChannelId(1)) -> ChannelId(0):
        self.transition_to(InvariantOperator.loop)
        self.value = value
        return value

    def clear(self, config: Configuration):
        self.value = None
        self.transition_to(InvariantOperator.start)

    def loop(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), InvariantOperator.invariant, InvariantOperator.clear)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), InvariantOperator.clear, InvariantOperator.invariant)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def invariant(self, config: Configuration) -> ChannelId(0):
        self.transition_to(InvariantOperator.loop)
        return self.value


@Operator.implement("STREAM_FU_CFG_T")
class StreamOperator(Operator):
    def __init__(self, *args):
        super().__init__(*args)

        self.current: Optional[smt.SMTTerm] = None
        self.end: Optional[smt.SMTTerm] = None

    def match(self, other: Operator) -> MatchingResult:
        result = super().match(other)
        assert isinstance(other, StreamOperator)

        if (self.current is None) == (other.current is None) == (self.end is None) == (other.end is None):
            if self.current is None:
                return result
            else:
                return result \
                    .merge(MatchingResult.match_smt_terms(self.current, other.current)) \
                    .merge(MatchingResult.match_smt_terms(self.end, other.end))

        else:
            return result.merge(MatchingFailure(f"stream state not matched at operator {self.pe.id}"))

    def copy(self) -> StreamOperator:
        copied = super().copy()
        copied.current = self.current
        copied.end = self.end
        return copied

    def start(self, config: Configuration, first: ChannelId(0), end: ChannelId(1)):
        self.current = first
        self.end = end
        # print("start", str(self.pe.id), self.current, self.end)
        self.transition_to(StreamOperator.loop)
        return ()

    def loop(self, config: Configuration) -> Branching:
        # print("loop", str(self.pe.id), self.current, self.end)
        return Branching(smt.BVSGE(self.current, self.end), StreamOperator.done, StreamOperator.not_done)

    def done(self, config: Configuration) -> ChannelId(1):
        self.current = None
        self.end = None
        self.transition_to(StreamOperator.start)

        # print("end", str(self.pe.id))

        if self.pe.pred == "STREAM_CFG_PRED_FALSE":
            return smt.BVConst(1, WORD_WIDTH)
        elif self.pe.pred == "STREAM_CFG_PRED_TRUE":
            return smt.BVConst(0, WORD_WIDTH)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def not_done(self, config: Configuration, step: ChannelId(2)) -> Tuple[ChannelId(0), ChannelId(1)]:
        current = self.current
        self.current = smt.BVAdd(self.current, step)
        self.transition_to(StreamOperator.loop)

        # print("not_end", str(self.pe.id))

        if self.pe.pred == "STREAM_CFG_PRED_FALSE":
            done_flag = smt.BVConst(0, WORD_WIDTH)
        elif self.pe.pred == "STREAM_CFG_PRED_TRUE":
            done_flag = smt.BVConst(1, WORD_WIDTH)
        else:
            assert False, f"unknown pred {self.pe.pred}"

        return current, done_flag


@Operator.implement("MEM_CFG_OP_STORE")
class StoreOperator(Operator):
    def __init__(self, *args):
        super().__init__(*args)

        if len(self.pe.inputs) == 3:
            self.transition_to(StoreOperator.start_3)

        elif len(self.pe.inputs) == 4:
            self.transition_to(StoreOperator.start_4)

        else:
            assert False, "unexpected number of input channels to the store operator"

    def is_at_start(self) -> bool:
        return self.current_transition == StoreOperator.start_3 or self.current_transition == StoreOperator.start_4

    def start_3(self, config: Configuration, base: ChannelId(0), index: ChannelId(1), value: ChannelId(2)) -> ChannelId(0):
        config.write_memory(base, index, value)
        return smt.BVConst(1, WORD_WIDTH)

    def start_4(self, config: Configuration, base: ChannelId(0), index: ChannelId(1), value: ChannelId(2), sync: ChannelId(3)) -> ChannelId(0):
        config.write_memory(base, index, value)
        return smt.BVConst(1, WORD_WIDTH)


@Operator.implement("MEM_CFG_OP_LOAD")
class LoadOperator(Operator):
    def __init__(self, *args):
        super().__init__(*args)

        if len(self.pe.inputs) == 2:
            self.transition_to(LoadOperator.start_2)

        elif len(self.pe.inputs) == 3:
            self.transition_to(LoadOperator.start_3)

        else:
            assert False, "unexpected number of input channels to the store operator"

    def is_at_start(self) -> bool:
        return self.current_transition == LoadOperator.start_2 or self.current_transition == LoadOperator.start_3

    def start_2(self, config: Configuration, base: ChannelId(0), index: ChannelId(1)) -> ChannelId(0):
        return config.read_memory(base, index)

    def start_3(self, config: Configuration, base: ChannelId(0), index: ChannelId(1), sync: ChannelId(2)) -> ChannelId(0):
        return config.read_memory(base, index)


# Only to be used in patterns
class WildcardOperator(Operator):
    def start(self, config: Configuration):
        assert False, "cannot execute the wildcard operator"


@dataclass
class PermissionedValue:
    term: smt.SMTTerm
    permission: permission.Variable

    def __str__(self) -> str:
        return str(self.term)


@dataclass
class ChannelState:
    hold_constant: Optional[PermissionedValue] = None

    # 0 is the head, -1 is the tail
    values: List[PermissionedValue] = field(default_factory=list)

    def pop(self) -> PermissionedValue:
        if self.hold_constant is not None:
            return self.hold_constant

        assert len(self.values) != 0, "popping an empty channel"
        return self.values.pop(0)

    def peek(self) -> PermissionedValue:
        if self.hold_constant is not None:
            return self.hold_constant

        assert len(self.values) != 0, "peeking an empty channel"
        return self.values[0]

    def push(self, value: PermissionedValue) -> None:
        assert self.hold_constant is None, "pushing into a constant channel"
        self.values.append(value)

    def ready(self) -> bool:
        return self.hold_constant is not None or len(self.values) != 0

    def count(self) -> int:
        assert self.hold_constant is None
        return len(self.values)

    def copy(self) -> ChannelState:
        return ChannelState(self.hold_constant, list(self.values))


# Only to be used in patterns
@dataclass
class WildcardChannelState(ChannelState):
    def pop(self) -> PermissionedValue:
        assert False, "cannot pop from a wildcard channel state"

    def peek(self) -> PermissionedValue:
        assert False, "cannot peek a wildcard channel state"

    def push(self, value: PermissionedValue) -> None:
        assert False, "cannot push to a wildcard channel state"

    def ready(self) -> bool:
        assert False, "cannot check status of a wildcard channel state"

    def count(self) -> int:
        assert False, "cannot count values in a wildcard channel state"

    def copy(self) -> ChannelState:
        return self


@dataclass
class MemoryUpdate:
    base: smt.SMTTerm
    index: smt.SMTTerm
    value: smt.SMTTerm

    def __str__(self) -> str:
        return f"{self.base}[{self.index}] = {self.value}"


class StepResult:
    ...


@dataclass
class NextConfiguration(StepResult):
    config: Configuration


@dataclass
class StepException(StepResult):
    reason: str


class InspectCache:
    method_to_signature: Dict[Callable, Any] = {}

    @staticmethod
    def signature(method: Callable) -> Any:
        if method not in InspectCache.method_to_signature:
            InspectCache.method_to_signature[method] = inspect.signature(method, eval_str=True)
        return InspectCache.method_to_signature[method]


@dataclass
class Configuration:
    graph: DataflowGraph
    free_vars: Mapping[str, smt.SMTTerm]

    operator_states: Tuple[Operator, ...] = ()
    channel_states: Tuple[ChannelState, ...] = ()

    # Memory is currently modelled as a map
    # address (WORD_WIDTH) |-> value (WORD_WIDTH)
    memory_updates: List[MemoryUpdate] = field(default_factory=list)
    memory: smt.SMTTerm = field(default_factory=lambda: Configuration.get_fresh_memory_var())

    path_conditions: List[smt.SMTTerm] = field(default_factory=list)

    permission_constraints: List[permission.Formula] = field(default_factory=list)

    solver: Optional[smt.Solver] = None

    disable_permissions: bool = False

    def match(self, other: Configuration) -> Tuple[MatchingResult, Optional[Tuple[permission.Equality, ...]]]:
        """
        Treat self as a pattern and match other against self

        Assumption on self:
        - self.graph == other.graph
        - free_var is empty
        - memory_updates is empty

        An additional list of permission equality is returned
        (and it will always be the case that for any equality left = right, left occurs in self and right occurs in other)
        """

        assert self.graph == other.graph
        # assert len(self.free_vars) == 0
        assert len(self.memory_updates) == 0

        result = MatchingSuccess()

        permission_equalities: List[permission.Equality] = []

        # Match operator states
        assert len(self.operator_states) == len(other.operator_states)
        for self_op, other_op in zip(self.operator_states, other.operator_states):
            if isinstance(self_op, WildcardOperator):
                ...
                # TODO: do anything?

            else:
                assert not isinstance(other_op, WildcardOperator)
                result = result.merge(self_op.match(other_op))
                if isinstance(result, MatchingFailure):
                    return result, None

                if self_op.internal_permission.name != other_op.internal_permission.name:
                    permission_equalities.append(permission.Equality(self_op.internal_permission, other_op.internal_permission))

        # Match channel states
        assert len(self.channel_states) == len(other.channel_states)
        for i, (self_channel, other_channel) in enumerate(zip(self.channel_states, other.channel_states)):
            if isinstance(self_channel, WildcardChannelState):
                ...
                # TODO: do anything here?

            else:
                assert not isinstance(other_channel, WildcardChannelState)

                if self_channel.hold_constant is None != other_channel.hold_constant is None:
                    return MatchingFailure(f"unmatched hold constant at channel {i}"), None

                if self_channel.hold_constant is None:
                    if len(self_channel.values) != len(other_channel.values):
                        return MatchingFailure(f"unmatched channel queue length at channel {i} ({len(self_channel.values)} vs {len(other_channel.values)})"), None
                    else:
                        for self_value, other_value in zip(self_channel.values, other_channel.values):
                            result = result.merge(MatchingResult.match_smt_terms(self_value.term, other_value.term))
                            if isinstance(result, MatchingFailure):
                                return result, None
                            if self_value.permission.name != other_value.permission.name:
                                permission_equalities.append(permission.Equality(self_value.permission, other_value.permission))
                else:
                    result = result.merge(MatchingSuccess(condition=smt.Equals(
                        self_channel.hold_constant.term,
                        other_channel.hold_constant.term,
                    )))
                    if self_channel.hold_constant.permission.name != other_channel.hold_constant.permission.name:
                        permission_equalities.append(permission.Equality(self_channel.hold_constant.permission, other_channel.hold_constant.permission))
                    if isinstance(result, MatchingFailure):
                        return result, None

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
        )), tuple(permission_equalities)

    @staticmethod
    def get_initial_permission_constraints(config: Configuration) -> List[permission.Formula]:
        initial_permissions: List[permission.Variable] = []
        hold_permissions: List[permission.Variable] = []

        for operator_state in config.operator_states:
            initial_permissions.append(operator_state.internal_permission)

        for channel_state in config.channel_states:
            if channel_state.hold_constant is not None:
                initial_permissions.append(channel_state.hold_constant.permission)
                hold_permissions.append(channel_state.hold_constant.permission)
            else:
                for value in channel_state.values:
                    initial_permissions.append(value.permission)

        # All initial permissions have to be disjoint
        permission_constraints = [ permission.Disjoint(tuple(initial_permissions)) ]

        # Hold permissions should be disjoint from itself (i.e. empty or read)
        for perm in hold_permissions:
            permission_constraints.append(permission.Disjoint((perm, perm)))

        return permission_constraints

    @staticmethod
    def get_initial_configuration(
        graph: DataflowGraph,
        free_vars: Mapping[str, smt.SMTTerm],
        solver: Optional[smt.Solver] = None,
        permission_prefix: str = "",
        disable_permissions: bool = False,
    ) -> Configuration:
        config = Configuration(graph, free_vars, solver=solver, disable_permissions=disable_permissions)

        # Initialize operator implementations
        operator_states: List[Operator] = []
        for i, vertex in enumerate(graph.vertices):
            assert vertex.operator in Operator.OPERATOR_IMPL_MAP, \
                   f"unable to find an implementation for operator `{vertex.operator}`"
            impl = Operator.OPERATOR_IMPL_MAP[vertex.operator]
            perm_var = config.get_fresh_permission_var(f"{permission_prefix}{PERMISSION_PREFIX_INTERNAL}{i}-")
            operator_states.append(impl(vertex, perm_var))
        config.operator_states = tuple(operator_states)

        # Initialize channel states
        channel_states: List[ChannelState] = []
        for channel in config.graph.channels:
            if channel.constant is not None:
                const_or_hold = PERMISSION_PREFIX_HOLD_CONSTANT if channel.hold else PERMISSION_PREFIX_CONSTANT

                if isinstance(channel.constant, ConstantValue):
                    value = config.get_fresh_permissioned_value(smt.BVConst(channel.constant.value, WORD_WIDTH), f"{permission_prefix}{const_or_hold}{channel.id}-")

                else:
                    assert isinstance(channel.constant, FunctionArgument)
                    name = channel.constant.variable_name
                    assert name in config.free_vars, f"unable to find an assignment to free var {name}"
                    value = config.get_fresh_permissioned_value(config.free_vars[name], f"{permission_prefix}{const_or_hold}{channel.id}-")

                if channel.hold:
                    state = ChannelState(value)
                else:
                    state = ChannelState()
                    state.push(value)

            else:
                # normal, empty channel
                state = ChannelState()

            channel_states.append(state)

        config.channel_states = tuple(channel_states)

        if not disable_permissions:
            config.permission_constraints = list(Configuration.get_initial_permission_constraints(config))

        return config

    @staticmethod
    def get_fresh_memory_var() -> smt.SMTTerm:
        return smt.FreshSymbol(smt.ArrayType(smt.BVType(WORD_WIDTH), smt.BVType(WORD_WIDTH)), "dataflow_mem_%d")

    def get_free_permission_vars(self) -> Tuple[permission.Variable, ...]:
        """
        Get all permission variables in the configuration (not including permission constraints)
        """
        permission_vars: List[permission.Variable] = []

        for operator_state in self.operator_states:
            permission_vars.append(operator_state.internal_permission)

        for channel_state in self.channel_states:
            if channel_state.hold_constant is not None:
                permission_vars.append(channel_state.hold_constant.permission)

            for value in channel_state.values:
                permission_vars.append(value.permission)

        return tuple(permission_vars)

    def copy(self) -> Configuration:
        return Configuration(
            self.graph,
            self.free_vars,
            tuple(operator.copy() for operator in self.operator_states),
            tuple(state.copy() for state in self.channel_states),
            list(self.memory_updates),
            self.memory,
            list(self.path_conditions),
            list(self.permission_constraints),
            self.solver,
            self.disable_permissions,
        )

    def write_memory(self, base: smt.SMTTerm, index: smt.SMTTerm, value: smt.SMTTerm):
        self.memory_updates.append(MemoryUpdate(base, index, value))
        self.memory = smt.Store(self.memory, smt.BVAdd(base, index), value)

    def read_memory(self, base: smt.SMTTerm, index: smt.SMTTerm) -> smt.SMTTerm:
        return smt.Select(self.memory, smt.BVAdd(base, index))

    def get_fresh_permission_var(self, prefix: str) -> permission.Variable:
        return permission.GlobalPermissionVarCounter.get_fresh_permission_var(prefix)

    def get_fresh_permissioned_value(self, value: smt.SMTTerm, prefix: str) -> PermissionedValue:
        return PermissionedValue(value, self.get_fresh_permission_var(prefix))

    def check_feasibility(self) -> bool:
        return smt.check_sat(self.path_conditions, self.solver)

    @staticmethod
    def get_transition_input_channels(pe_info: ProcessingElement, transition: TransitionFunction) -> Tuple[int, ...]:
        signature = InspectCache.signature(transition)
        channel_ids = []

        for i, (name, channel_param) in enumerate(signature.parameters.items()):
            if i == 0:
                assert name == "self", f"ill-formed first transition parameter {channel_param}"
            elif i == 1:
                assert channel_param.annotation is Configuration, f"ill-formed second transition parameter {channel_param}"
            else:
                assert isinstance(channel_param.annotation, ChannelId), f"ill-formed transition parameter {channel_param}"
                channel_id = pe_info.inputs[channel_param.annotation.id].id
                channel_ids.append(channel_id)

        return tuple(channel_ids)

    @staticmethod
    def get_transition_output_channels(pe_info: ProcessingElement, transition: TransitionFunction) -> Tuple[int, ...]:
        channel_ids = []

        for action in Configuration.get_transition_output_actions(transition):
            if isinstance(action, ChannelId) and action.id in pe_info.outputs:
                for output_channel in pe_info.outputs[action.id]:
                    channel_ids.append(output_channel.id)

        return tuple(channel_ids)

    @staticmethod
    def get_transition_output_actions(transition: TransitionFunction) -> Tuple[Any, ...]:
        return_annotation = InspectCache.signature(transition).return_annotation

        if return_annotation is Branching:
            return return_annotation,

        elif isinstance(return_annotation, ChannelId):
            return return_annotation,

        elif return_annotation is inspect.Signature.empty:
            return ()

        elif "__origin__" in dir(return_annotation) and return_annotation.__origin__ is tuple:
            return tuple(return_annotation.__args__)

        else:
            assert False, f"unrecognized transition return annotation {return_annotation}"

    def __str__(self) -> str:
        lines: List[str] = []

        lines.append("operator states:")

        for operator_state in self.operator_states:
            lines.append(f"  {operator_state.pe.id}: {operator_state.__class__.__name__}@{operator_state.current_transition.__name__}")

        lines.append("channel states:")

        for id, channel_state in enumerate(self.channel_states):
            if channel_state.hold_constant is not None:
                lines.append(f"  {id}: (hold) {channel_state.hold_constant}")
            else:
                if len(channel_state.values) == 0:
                    lines.append(f"  {id}: (empty)")
                else:
                    lines.append(f"  {id}:")
                    for value in channel_state.values:
                        lines.append(f"    {value}")

        lines.append(f"memory updates:")
        for update in self.memory_updates:
            lines.append(f"  {update}")

        lines.append(f"path conditions:")
        for constraint in self.path_conditions:
            lines.append(f"  {constraint}")

        # max_line_width = max(map(len, lines))
        # lines = [ f"## {line}{' ' * (max_line_width - len(line))} ##" for line in lines ]
        # lines = [ "#" * (max_line_width + 6) ] + lines + [ "#" * (max_line_width + 6) ]

        lines = [ "|| " + line for line in lines ]
        lines = ["===== dataflow state begin ====="] + lines + ["===== dataflow state end ====="]

        return "\n".join(lines)

    def step_exhaust(self, pe_id: int, stop_at_start: bool = True, **kwargs) -> Tuple[StepResult, ...]:
        """
        Similar to step, but will attempt all possible transitions on the specified PE,
        including those after branching.

        when stop_at_start is true, the execution will also stop if the PE hit the start transition again

        Note that this may not terminate on some PEs such as Stream
        """

        final_results = []
        results: List[NextConfiguration] = list(self.step(pe_id, **kwargs))

        while len(results) != 0:
            result = results.pop(0)

            if result.config.operator_states[pe_id].is_at_start():
                final_results.append(result)
                continue

            if isinstance(result, NextConfiguration):
                next_results = result.config.step(pe_id)

                if len(next_results) == 0:
                    final_results.append(result)
                else:
                    results.extend(next_results)

            elif isinstance(result, StepException):
                final_results.append(result)

            else:
                assert False, f"unexpected step result {result}"

        return tuple(final_results)

    def is_fireable(self, pe_id: int) -> bool:
        pe_info = self.graph.vertices[pe_id]
        operator_state = self.operator_states[pe_info.id]
        transition = operator_state.current_transition
        input_channel_ids = Configuration.get_transition_input_channels(pe_info, transition)

        # Check for input channel availability
        # print(f"{pe_info.id} {input_channel_ids}")
        for channel_id in input_channel_ids:
            if not self.channel_states[channel_id].ready():
                # Channel not ready
                return False

        # Check for output channel capacity
        for channel_id in Configuration.get_transition_output_channels(pe_info, transition):
            bound = self.graph.channels[channel_id].bound
            if bound is not None and self.channel_states[channel_id].count() >= bound:
                return False

        return True

    def is_final(self) -> bool:
        for pe in self.graph.vertices:
            if self.is_fireable(pe.id):
                return False
        return True

    def step(self, pe_id: int, base_pointer_mapping: Optional[Mapping[str, str]] = None) -> Tuple[StepResult, ...]:
        """
        Execute the `pe_index`th PE in the dataflow graph for at most one transition
        Returns () if no transition is possible,
        Otherwise return a tuple of results
        """

        pe_info = self.graph.vertices[pe_id]
        operator_state = self.operator_states[pe_info.id]
        transition = operator_state.current_transition
        input_channel_ids = Configuration.get_transition_input_channels(pe_info, transition)

        if not self.is_fireable(pe_id):
            return ()

        # TODO: when channels are bounded, check for output channel availability here

        # Pop values from the input channels
        input_permissions: List[permission.Variable] = [operator_state.internal_permission]
        input_values = []
        for channel_id in input_channel_ids:
            value = self.channel_states[channel_id].pop()
            input_permissions.append(value.permission)
            input_values.append(value.term)

        # Run the transition
        # print(f"original transition {operator_state.current_transition}")
        output_values = transition(operator_state, self, *input_values)
        # print(f"final transition {operator_state.current_transition} {self.get_transition_input_channels(pe_info, operator_state.current_transition)}")

        # Process output behaviors
        output_actions = Configuration.get_transition_output_actions(transition)

        if len(output_actions) == 0:
            output_values = ()

        elif len(output_actions) == 1 and not isinstance(output_values, tuple):
            output_values = output_values,

        assert len(output_values) == len(output_actions), f"output of transition {transition} does not match its specification"

        # If the operation is a store or load
        # Add additional permission constraint of read/write A <= input permissions
        if not self.disable_permissions:
            if isinstance(operator_state, LoadOperator) or isinstance(operator_state, StoreOperator):
                base_pointers = permission.PermissionSolver.find_function_argument_producers(self.graph, pe_info.inputs[0].id)

                unique_base_pointers = []

                # Some pointers not marked with restrict keyword needs to be put together
                for base_pointer in base_pointers:
                    if base_pointer_mapping is not None:
                        base_pointer = base_pointer_mapping[base_pointer]
                    if base_pointer not in unique_base_pointers:
                        unique_base_pointers.append(base_pointer)

                if len(unique_base_pointers) == 0:
                    assert False, "unsupported permission for store/load"

                if isinstance(operator_state, LoadOperator):
                    # Require at least one read permission for each base pointer
                    for base_pointer in unique_base_pointers:
                        self.permission_constraints.append(permission.HasRead(base_pointer, permission.DisjointUnion(input_permissions)))

                else:
                    self.permission_constraints.append(permission.Inclusion(
                        permission.DisjointUnion.of(*(
                            permission.Write(base_pointer)
                            for base_pointer in unique_base_pointers
                        )),
                        permission.DisjointUnion(input_permissions),
                    ))

        # Update internal permission
        operator_state.internal_permission = self.get_fresh_permission_var(f"{PERMISSION_PREFIX_EXEC}{PERMISSION_PREFIX_INTERNAL}{pe_info.id}-")
        output_permissions: List[permission.Variable] = [operator_state.internal_permission]

        if len(output_actions) == 1 and output_actions[0] is Branching:
            # A single branching action
            output_value = output_values[0]
            assert isinstance(output_value, Branching), f"output of transition {transition} does not match its specification"

            configuration_true = self
            configuration_false = self.copy()

            configuration_true.operator_states[pe_info.id].current_transition = output_value.true_branch
            configuration_true.path_conditions.append(output_value.condition.simplify())

            configuration_false.operator_states[pe_info.id].current_transition = output_value.false_branch
            configuration_false.path_conditions.append(smt.Not(output_value.condition).simplify())

            if not self.disable_permissions:
                permission_constraint = permission.Inclusion(
                    permission.DisjointUnion.of(*output_permissions),
                    permission.DisjointUnion.of(*input_permissions),
                )
                configuration_true.permission_constraints.append(permission_constraint)
                configuration_false.permission_constraints.append(permission_constraint)

            if not configuration_true.check_feasibility():
                # since the negation is unsat, the condition itself must be valid
                configuration_false.path_conditions.pop()
                return NextConfiguration(configuration_false),

            if not configuration_false.check_feasibility():
                configuration_true.path_conditions.pop()
                return NextConfiguration(configuration_true),

            return NextConfiguration(configuration_true), NextConfiguration(configuration_false)
        else:
            # Output channels
            for value, action in zip(output_values, output_actions):
                assert isinstance(action, ChannelId), f"unexpected action {action}"
                if action.id not in pe_info.outputs:
                    continue

                channels = pe_info.outputs[action.id]
                for channel in channels:
                    permissioned_value = self.get_fresh_permissioned_value(value.simplify(), f"{PERMISSION_PREFIX_EXEC}{PERMISSION_PREFIX_OUTPUT_CHANNEL}{channel.id}-")
                    self.channel_states[channel.id].push(permissioned_value)
                    output_permissions.append(permissioned_value.permission)

            # Add permission constraint:
            #   output_permission + new_internal_permission <= input_permission + old_internal_permission
            if not self.disable_permissions:
                permission_constraint = permission.Inclusion(
                    permission.DisjointUnion.of(*output_permissions),
                    permission.DisjointUnion.of(*input_permissions),
                )

                self.permission_constraints.append(permission_constraint)

            return NextConfiguration(self),

    def step_until_branch(self, pe_ids: Iterable[int], exhaust: bool = True, **kwargs) -> Tuple[StepResult, ...]:
        """
        Run the specified PEs in sequence and return immediately if branching happens.
        If all given PEs are not fireable, return ()
        Otherwise return a single NextConfiguration
        """

        updated = False

        for pe_id in pe_ids:
            results = self.step_exhaust(pe_id, **kwargs) if exhaust else self.step(pe_id)

            if len(results) == 1:
                assert isinstance(results[0], NextConfiguration)
                self = results[0].config
                updated = True

            elif len(results) > 1:
                # branching, return immediately
                return results

            # otherwise if no step, continue

        if updated:
            return NextConfiguration(self),

        return ()
