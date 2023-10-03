from __future__ import annotations

from typing import Tuple, Optional, List, Generator, Type, Dict, Mapping, Any, Set, Callable
from dataclasses import dataclass, field

import inspect

import semantics.smt as smt

from . import permission
from .graph import DataflowGraph, ProcessingElement, ConstantValue, FunctionArgument


WORD_WIDTH = 64


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
    
    def __init__(self, pe: ProcessingElement, permission: permission.PermissionVariable, transition: Optional[TransitionFunction] = None):
        self.pe = pe
        self.internal_permission: permission.PermissionVariable = permission
        self.current_transition: TransitionFunction = transition or type(self).start

    def transition_to(self, transition: TransitionFunction):
        self.current_transition = transition

    def start(self, config: Configuration):
        raise NotImplementedError()
    
    def copy(self) -> Operator:
        return type(self)(self.pe, self.internal_permission, self.current_transition)


@Operator.implement("ARITH_CFG_OP_EQ")
class EqOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.Equals(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_ADD")
class AddOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        # print("plus", a, b)
        return smt.BVAdd(a, b)
    

@Operator.implement("MUL_CFG_OP_MUL")
class MulOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        # print("mul", a, b)
        return smt.BVMul(a, b)
    

@Operator.implement("ARITH_CFG_OP_SGT")
class SignedGreaterThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVSGT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_UGT")
class UnsignedGreaterThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVUGT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("ARITH_CFG_OP_ULT")
class UnsignedLessThanOperator(Operator):
    def start(self, config: Configuration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.BVULT(a, b), smt.BVConst(1, WORD_WIDTH), smt.BVConst(0, WORD_WIDTH))


@Operator.implement("CF_CFG_OP_SELECT")
class SelectOperator(Operator):
    def start(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), SelectOperator.false, SelectOperator.true)
    
    def true(self, config: Configuration, a: ChannelId(1), b: ChannelId(2)) -> ChannelId(0):
        return a

    def false(self, config: Configuration, a: ChannelId(1), b: ChannelId(2)) -> ChannelId(0):
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

    def start(self, config: Configuration, value: ChannelId(1)) -> ChannelId(0):
        self.transition_to(InvariantOperator.loop)
        self.value = value
        return value
    
    def loop(self, config: Configuration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), InvariantOperator.invariant, InvariantOperator.start)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.BVConst(0, WORD_WIDTH)), InvariantOperator.start, InvariantOperator.invariant)
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

    def start_2(self, config: Configuration, base: ChannelId(0), index: ChannelId(1)) -> ChannelId(0):
        return config.read_memory(base, index)

    def start_3(self, config: Configuration, base: ChannelId(0), index: ChannelId(1), sync: ChannelId(2)) -> ChannelId(0):
        return config.read_memory(base, index)


@dataclass
class PermissionedValue:
    term: smt.SMTTerm
    permission: permission.PermissionVariable

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

    def push(self, value: PermissionedValue) -> None:
        assert self.hold_constant is None, "pushing into a constant channel"
        self.values.append(value)

    def ready(self) -> bool:
        return self.hold_constant is not None or len(self.values) != 0

    def copy(self) -> ChannelState:
        return ChannelState(self.hold_constant, list(self.values))


@dataclass
class MemoryUpdate:
    base: smt.SMTTerm
    index: smt.SMTTerm
    value: smt.SMTTerm


class StepResult:
    ...


@dataclass
class NextConfiguration(StepResult):
    config: Configuration


@dataclass
class NoTransition(StepResult):
    ...


@dataclass
class StepException(StepResult):
    reason: str


@dataclass
class Configuration:
    graph: DataflowGraph
    free_vars: Mapping[str, smt.SMTTerm]

    operator_states: Tuple[Operator, ...] = ()
    channel_states: Tuple[ChannelState, ...] = ()

    # Memory is currently modelled as a map
    # base |-> array of ints
    # This is assuming all bases represent disjoint regions of memory
    memory: List[MemoryUpdate] = field(default_factory=list)
    memory_var: smt.SMTTerm = field(default_factory=lambda: Configuration.get_fresh_memory_var())
    memory_constraints: List[smt.SMTTerm] = field(default_factory=list)

    path_constraints: List[smt.SMTTerm] = field(default_factory=list)
    permission_constraints: List[permission.Formula] = field(default_factory=list)

    permission_var_count: int = 0

    @staticmethod
    def get_initial_configuration(graph: DataflowGraph, free_vars: Mapping[str, smt.SMTTerm]):
        initial_permissions: List[permission.PermissionVariable] = []
        hold_permissions: List[permission.PermissionVariable] = []

        config = Configuration(graph, free_vars)

        # Initialize operator implementations
        operator_states: List[Operator] = []
        for i, vertex in enumerate(graph.vertices):
            assert vertex.operator in Operator.OPERATOR_IMPL_MAP, \
                   f"unable to find an implementation for operator {vertex.operator}"
            impl = Operator.OPERATOR_IMPL_MAP[vertex.operator]
            perm_var = config.get_fresh_permission_var(f"internal-{i}-")
            initial_permissions.append(perm_var)
            operator_states.append(impl(vertex, perm_var))
        config.operator_states = tuple(operator_states)

        # Initialize channel states
        channel_states: List[ChannelState] = []
        for channel in config.graph.channels:
            if channel.constant is not None:
                if isinstance(channel.constant, ConstantValue):
                    value = config.get_fresh_permissioned_value(smt.BVConst(channel.constant.value, WORD_WIDTH))
                
                else:
                    assert isinstance(channel.constant, FunctionArgument)
                    name = channel.constant.variable_name
                    assert name in config.free_vars, f"unable to find an assignment to free var {name}"
                    value = config.get_fresh_permissioned_value(config.free_vars[name])

                initial_permissions.append(value.permission)

                if channel.hold:
                    state = ChannelState(value)
                    hold_permissions.append(value.permission)
                
                else:
                    state = ChannelState()
                    state.push(value)

            else:
                # normal, empty channel
                state = ChannelState()

            channel_states.append(state)
        config.channel_states = tuple(channel_states)

        # All initial permissions have to be disjoint
        config.permission_constraints.append(permission.Disjoint(tuple(initial_permissions)))

        # Hold permissions should be disjoint from itself (i.e. empty or read)
        for perm in hold_permissions:
            config.permission_constraints.append(permission.Disjoint((perm, perm)))

        return config

    @staticmethod
    def get_fresh_memory_var() -> smt.SMTTerm:
        return smt.FreshSymbol(smt.ArrayType(smt.BVType(WORD_WIDTH), smt.ArrayType(smt.BVType(WORD_WIDTH), smt.BVType(WORD_WIDTH))))

    def copy(self) -> Configuration:
        return Configuration(
            self.graph,
            self.free_vars,
            tuple(operator.copy() for operator in self.operator_states),
            tuple(state.copy() for state in self.channel_states),
            list(self.memory),
            self.memory_var,
            list(self.memory_constraints),
            list(self.path_constraints),
            list(self.permission_constraints),
            self.permission_var_count,
        )
    
    def write_memory(self, base: smt.SMTTerm, index: smt.SMTTerm, value: smt.SMTTerm):
        self.memory.append(MemoryUpdate(base, index, value))

        new_var = Configuration.get_fresh_memory_var()
        self.memory_constraints.append(smt.Equals(
            new_var,
            smt.Store(self.memory_var, base, smt.Store(smt.Select(self.memory_var, base), index, value)),
        ))
        self.memory_var = new_var

    def read_memory(self, base: smt.SMTTerm, index: smt.SMTTerm) -> smt.SMTTerm:
        return smt.Select(smt.Select(self.memory_var, base), index)

    def get_fresh_permission_var(self, prefix="p") -> permission.PermissionVariable:
        var = permission.PermissionVariable(f"{prefix}{self.permission_var_count}")
        self.permission_var_count += 1
        return var

    def get_fresh_permissioned_value(self, value: smt.SMTTerm) -> PermissionedValue:
        return PermissionedValue(value, self.get_fresh_permission_var())

    def check_feasibility(self) -> bool:
        with smt.Solver(name="z3") as solver:
            for path_condition in self.path_constraints:
                solver.add_assertion(path_condition)

            # true for sat, false for unsat
            return solver.solve()

    def get_transition_input_channels(self, pe_info: ProcessingElement, transition: TransitionFunction) -> Tuple[int, ...]:
        signature = inspect.signature(transition, eval_str=True)
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
    
    def get_transition_output_actions(self, transition: TransitionFunction) -> Tuple[Any, ...]:
        return_annotation = inspect.signature(transition, eval_str=True).return_annotation

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
        for update in self.memory:
            lines.append(f"  {update.base}[{update.index}] = {update.value}")

        lines.append(f"path conditions:")
        for constraint in self.path_constraints:
            lines.append(f"  {constraint}")

        max_line_width = max(map(len, lines))
        lines = [ f"## {line}{' ' * (max_line_width - len(line))} ##" for line in lines ]
        lines = [ "#" * (max_line_width + 6) ] + lines + [ "#" * (max_line_width + 6) ]

        return "\n".join(lines)

    def step_exhaust(self, pe_id: int) -> Tuple[StepResult, ...]:
        """
        Similar to step, but will attempt all possible transitions on the specified PE,
        including those after branching
        """

        final_results = []
        results = list(self.step(pe_id))

        if len(results) == 1 and isinstance(results[0], NoTransition):
            return NoTransition(),

        while len(results) != 0:
            result = results.pop(0)

            if isinstance(result, NextConfiguration):
                next_results = result.config.step(pe_id)

                if len(next_results) == 1 and isinstance(next_results[0], NoTransition):
                    final_results.append(next_results[0])
                else:
                    results.extend(next_results)

            elif isinstance(result, StepException):
                final_results.append(result)
            
            else:
                assert False, f"unexpected step result {result}"

        return final_results

    def step(self, pe_id: int) -> Tuple[StepResult, ...]:
        """
        Execute the `pe_index`th PE in the dataflow graph,
        until either one fo the following case happens:
        - A transition is executed
        - No transition possible
        """
        
        pe_info = self.graph.vertices[pe_id]
        operator_state = self.operator_states[pe_info.id]

        transition = operator_state.current_transition
        
        input_channel_ids = self.get_transition_input_channels(pe_info, transition)

        # Check for input channel availability
        # print(f"{pe_info.id} {input_channel_ids}")
        for channel_id in input_channel_ids:
            if not self.channel_states[channel_id].ready():
                # Channel not ready
                return NoTransition(),

        # print("triggering", str(pe_info.id))

        # TODO: when channels are bounded, check for output channel availability here

        # Pop values from the input channels
        input_permissions: List[permission.PermissionVariable] = [operator_state.internal_permission]
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
        output_actions = self.get_transition_output_actions(transition)

        if len(output_actions) == 0:
            output_values = ()

        elif len(output_actions) == 1 and not isinstance(output_values, tuple):
            output_values = output_values,

        assert len(output_values) == len(output_actions), f"output of transition {transition} does not match its specification"

        # If the operation is a store or load
        # Add additional permission constraint of read/write A <= input permissions
        if isinstance(operator_state, LoadOperator) or isinstance(operator_state, StoreOperator):
            # TODO: right now we assume that the base is always one of the free variables
            assert isinstance(pe_info.inputs[0].constant, FunctionArgument)
            var_name = pe_info.inputs[0].constant.variable_name
            assert var_name in self.free_vars

            if isinstance(operator_state, LoadOperator):
                mem_permission = permission.ReadPermission(var_name)
            else:
                mem_permission = permission.WritePermission(var_name)
            
            self.permission_constraints.append(permission.Inclusion(mem_permission, permission.DisjointUnion(input_permissions)))

        # Update internal permission
        operator_state.internal_permission = self.get_fresh_permission_var(f"internal-{pe_info.id}-")
        output_permissions: List[permission.PermissionVariable] = [operator_state.internal_permission]

        if len(output_actions) == 1 and output_actions[0] is Branching:
            # A single branching action
            output_value = output_values[0]
            assert isinstance(output_value, Branching), f"output of transition {transition} does not match its specification"

            configuration_true = self
            configuration_false = self.copy()

            permission_constraint = permission.Inclusion(
                permission.DisjointUnion.of(*output_permissions),
                permission.DisjointUnion.of(*input_permissions),
            )

            configuration_true.operator_states[pe_info.id].current_transition = output_value.true_branch
            configuration_true.path_constraints.append(output_value.condition.simplify())
            configuration_true.permission_constraints.append(permission_constraint)

            configuration_false.operator_states[pe_info.id].current_transition = output_value.false_branch
            configuration_false.path_constraints.append(smt.Not(output_value.condition).simplify())
            configuration_false.permission_constraints.append(permission_constraint)

            if not configuration_true.check_feasibility():
                # since the negation is unsat, the condition itself must be valid
                configuration_false.path_constraints.pop(-1)
                return NextConfiguration(configuration_false),

            if not configuration_false.check_feasibility():
                configuration_true.path_constraints.pop(-1)
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
                    permissioned_value = self.get_fresh_permissioned_value(value.simplify())
                    self.channel_states[channel.id].push(permissioned_value)
                    output_permissions.append(permissioned_value.permission)

            # Add permission constraint:
            #   output_permission + new_internal_permission <= input_permission + old_internal_permission
            permission_constraint = permission.Inclusion(
                permission.DisjointUnion.of(*output_permissions),
                permission.DisjointUnion.of(*input_permissions),
            )

            self.permission_constraints.append(permission_constraint)

            return NextConfiguration(self),
