from __future__ import annotations

from typing import Tuple, Optional, List, Generator, Type, Dict, Mapping, Any, Set, Callable
from dataclasses import dataclass, field

import smt
import permission

from dataflow import DataflowGraph, ProcessingElement, ConstantValue, FunctionArgument


class OperatorAction: ...


@dataclass
class PopChannelAction(OperatorAction):
    input_index: int


@dataclass
class PushChannelAction(OperatorAction):
    output_index: int
    value: smt.SMTTerm


@dataclass
class ReadMemoryAction(OperatorAction):
    base: smt.SMTTerm
    index: smt.SMTTerm


@dataclass
class WriteMemoryAction(OperatorAction):
    base: smt.SMTTerm
    index: smt.SMTTerm
    value: smt.SMTTerm


@dataclass
class BranchAction(OperatorAction):
    condition: smt.SMTTerm
    true_branch: Callable[[Operator], TransitionGenerator]
    false_branch: Callable[[Operator], TransitionGenerator]


TransitionGenerator = Generator[OperatorAction, Any, None]


class Operator:
    OPERATOR_IMPL_MAP: Dict[str, Type[Operator]] = {}

    @staticmethod
    def implement(name: str):
        def wrapper(cls: Type[Operator]):
            Operator.OPERATOR_IMPL_MAP[name] = cls
            return cls
        return wrapper
    
    def __init__(self, pe: ProcessingElement):
        self.pe = pe

    def transition(self) -> TransitionGenerator:
        raise NotImplementedError()
    
    def copy(self) -> Operator:
        return type(self)(self.pe)


@Operator.implement("ARITH_CFG_OP_ADD")
class AddOperator(Operator):
    def transition(self) -> TransitionGenerator:
        a = yield PopChannelAction(0)
        b = yield PopChannelAction(1)
        yield PushChannelAction(0, smt.Plus(a, b))


@Operator.implement("STREAM_FU_CFG_T")
class StreamOperator(Operator):
    def __init__(self, pe: ProcessingElement):
        super().__init__(pe)

        self.current: Optional[smt.SMTTerm] = None
        self.end: Optional[smt.SMTTerm] = None

    def transition(self) -> TransitionGenerator:
        if self.current is None:
            first = yield PopChannelAction(0)
            end = yield PopChannelAction(1)
            step = yield PopChannelAction(2)

            self.current = smt.Plus(first, step)
            self.end = end
            
            yield PushChannelAction(0, first)
            yield PushChannelAction(1, smt.FALSE())

        else:
            step = yield PopChannelAction(2)

            def true_branch(self) -> TransitionGenerator:
                yield PushChannelAction(1, smt.TRUE())
                self.current = None
                self.end = None

            def false_branch(self) -> TransitionGenerator:
                yield PushChannelAction(0, self.current)
                yield PushChannelAction(1, smt.FALSE())
                self.current = smt.Plus(self.current, step)

            yield BranchAction(smt.GE(self.current, self.end), true_branch, false_branch)

    def copy(self) -> StreamOperator:
        copied = super().copy()
        copied.current = self.current
        copied.end = self.end
        return copied


@Operator.implement("MEM_CFG_OP_STORE")
class StoreOperator(Operator):
    def transition(self) -> TransitionGenerator:
        base = yield PopChannelAction(0)
        index = yield PopChannelAction(1)
        value = yield PopChannelAction(2)

        # an additional synchronization signal
        if len(self.pe.inputs) == 4:
            yield PopChannelAction(3)

        yield WriteMemoryAction(base, index, value)
        yield PushChannelAction(0, smt.Int(1))


@dataclass
class PermissionedValue:
    term: smt.SMTTerm
    permission: permission.PermissionVariable


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
class SymbolicMemoryUpdate:
    base: smt.SMTTerm
    index: smt.SMTTerm
    value: smt.SMTTerm


@dataclass
class TransitionState:
    generator: TransitionGenerator
    transcript: List[SymbolicMemoryUpdate]

    def copy(self) -> TransitionState:
        return TransitionState(self.generator, list(self.transcript))


@dataclass
class SymbolicConfiguration:
    operator_states: Tuple[Operator, ...]
    channel_states: Tuple[ChannelState, ...]
    transition_states: List[Optional[TransitionState]]

    memory: List[SymbolicMemoryUpdate] = field(default_factory=list)
    permission_constraints: List[permission.Formula] = field(default_factory=list)

    path_conditions: List[smt.SMTTerm] = field(default_factory=list)

    def copy(self) -> SymbolicConfiguration:
        operator_states = tuple(operator.copy() for operator in self.operator_states)
        channel_states = tuple(state.copy() for state in self.channel_states)
        transition_states = list(state.copy() if state is not None else None for state in self.transition_states)
        memory = list(self.memory)
        permission_constraints = list(self.permission_constraints)
        path_conditions = list(self.path_conditions)

        return SymbolicConfiguration(
            operator_states,
            channel_states,
            transition_states,
            memory,
            permission_constraints,
            path_conditions,
        )


class SymbolicExecutor:
    def __init__(self, graph: DataflowGraph, free_vars: Mapping[str, smt.SMTTerm]):
        self.graph = graph

        self.free_vars = dict(free_vars)
        self.permission_var_count = 0

        # Initialize operator implementations
        operator_impl: List[Operator] = []
        for vertex in graph.vertices:
            assert vertex.operator in Operator.OPERATOR_IMPL_MAP, \
                   f"unable to find an implementation for operator {vertex.operator}"
            impl = Operator.OPERATOR_IMPL_MAP[vertex.operator]
            operator_impl.append(impl(vertex))

        # Initialize channel states
        channel_states: List[ChannelState] = []
        for channel in graph.channels:
            if channel.constant is not None:
                if isinstance(channel.constant, ConstantValue):
                    value = self.get_fresh_permissioned_value(smt.Int(channel.constant.value))
                
                else:
                    assert isinstance(channel.constant, FunctionArgument)
                    name = channel.constant.variable_name
                    assert name in free_vars, f"unable to find an assignment to free var {name}"
                    value = self.get_fresh_permissioned_value(free_vars[name])

                if channel.hold:
                    state = ChannelState(value)
                
                else:
                    state = ChannelState()
                    state.push(value)

            else:
                # normal, empty channel
                state = ChannelState()

            channel_states.append(state)

        # Initialize transition function states
        transition_states: List[TransitionGenerator] = [None] * len(operator_impl)

        init_config = SymbolicConfiguration(tuple(operator_impl), tuple(channel_states), transition_states)

        self.configurations: List[SymbolicConfiguration] = [init_config]

    def get_fresh_permissioned_value(self, value: smt.SMTTerm) -> PermissionedValue:
        return PermissionedValue(value, self.get_fresh_permission_var())

    def get_fresh_permission_var(self) -> permission.PermissionVariable:
        var = permission.PermissionVariable(f"p{self.permission_var_count}")
        self.permission_var_count += 1
        return var
    
    def check_branch_feasibility(self, configuration: SymbolicConfiguration) -> bool:
        with smt.Solver(name="z3") as solver:
            for path_condition in configuration.path_conditions:
                solver.add_assertion(path_condition)

            # true for sat, false for unsat
            return solver.solve()
    
    def step(self, configuration: SymbolicConfiguration) -> Tuple[SymbolicConfiguration, ...]:
        """
        Try to proceed with each transition function until one of the following happens:
        - All transitions have been tried for at least once
        - Branching
        """

        changed = False
        
        for pe_info, operator_state, transition_state in \
            zip(self.graph.vertices, configuration.operator_states, configuration.transition_states):
            if transition_state is not None:
                blocking_action = transition_state.transcript[-1]
                if isinstance(blocking_action, PopChannelAction):
                
                    channel_id = pe_info.inputs[blocking_action.input_index].id
                    
                    if configuration.channel_states[channel_id].ready():
                        transition_state.generator.send(configuration.channel_states[channel_id].pop().term)
                        changed = True

                    else:
                        # not ready to execute the transition yet
                        continue

                elif isinstance(blocking_action, BranchAction):
                    pass

                else:
                    assert False, f"unexpected blocking action {blocking_action}"

            else:
                transition_state = TransitionState(operator_state.transition(), [])
                configuration.transition_states[pe_info.id] = transition_state
                changed = True

            # print(pe_info)

            # try to continue executing the transition function
            send_value = None

            while True:
                try:
                    action = transition_state.generator.send(send_value)
                    transition_state.transcript.append(action)
                    send_value = None

                    if isinstance(action, PopChannelAction):
                        channel_id = pe_info.inputs[action.input_index].id
                        channel_state = configuration.channel_states[channel_id]
                        if channel_state.ready():
                            send_value = channel_state.pop().term
                        else:
                            # not ready, we are stuck on this operator
                            break

                    elif isinstance(action, PushChannelAction):
                        if action.output_index in pe_info.outputs:
                            channels = pe_info.outputs[action.output_index]
                            
                            for channel in channels:
                                channel_state = configuration.channel_states[channel.id]
                                channel_state.push(PermissionedValue(action.value, self.get_fresh_permission_var()))

                    elif isinstance(action, ReadMemoryAction):
                        raise NotImplementedError()

                    elif isinstance(action, WriteMemoryAction):
                        configuration.memory.append(SymbolicMemoryUpdate(action.base, action.index, action.value))

                    elif isinstance(action, BranchAction):
                        configuration_true = configuration
                        configuration_false = configuration.copy()

                        transition_state_true = TransitionState(action.true_branch(configuration_true.operator_states[pe_info.id]), transition_state.transcript)
                        configuration_true.transition_states[pe_info.id] = transition_state_true
                        configuration_true.path_conditions.append(action.condition)

                        transition_state_false = TransitionState(action.false_branch(configuration_false.operator_states[pe_info.id]), transition_state.transcript)
                        configuration_false.transition_states[pe_info.id] = transition_state_false
                        configuration_false.path_conditions.append(smt.Not(action.condition))

                        if not self.check_branch_feasibility(configuration_true):
                            configuration = configuration_false
                            transition_state = transition_state_false
                            continue

                        if not self.check_branch_feasibility(configuration_false):
                            configuration = configuration_true
                            transition_state = transition_state_true
                            continue

                        return configuration_true, configuration_false

                    else:
                        assert False, f"unsupported action {action}"
                
                except StopIteration:
                    # TODO: accumulate new permission constraints
                    configuration.transition_states[pe_info.id] = None
                    break

        if not changed:
            return ()

        return configuration,
