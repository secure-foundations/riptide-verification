from __future__ import annotations

from typing import Tuple, Optional, List, Generator, Type, Dict, Mapping, Any, Set, Callable
from dataclasses import dataclass, field

import inspect

import smt
import permission

from dataflow import DataflowGraph, ProcessingElement, ConstantValue, FunctionArgument


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
    
    def __init__(self, pe: ProcessingElement, transition: Optional[TransitionFunction] = None):
        self.pe = pe
        self.current_transition: TransitionFunction = transition or type(self).start

    def transition_to(self, transition: TransitionFunction):
        self.current_transition = transition

    def start(self, config: SymbolicConfiguration) -> TransitionGenerator:
        raise NotImplementedError()
    
    def copy(self) -> Operator:
        return type(self)(self.pe, self.current_transition)


@Operator.implement("ARITH_CFG_OP_ADD")
class AddOperator(Operator):
    def start(self, config: SymbolicConfiguration, a: ChannelId(0), b: ChannelId(1)) -> Tuple[ChannelId(0)]:
        return smt.Plus(a, b)


@Operator.implement("STREAM_FU_CFG_T")
class StreamOperator(Operator):
    def __init__(self, *args):
        super().__init__(*args)

        self.current: Optional[smt.SMTTerm] = None
        self.end: Optional[smt.SMTTerm] = None

    def start(self, config: SymbolicConfiguration, first: ChannelId(0), end: ChannelId(1)):
        self.current = first
        self.end = end
        self.transition_to(StreamOperator.loop)
        return ()

    def loop(self, config: SymbolicConfiguration) -> Branching:
        return Branching(smt.GE(self.current, self.end), StreamOperator.end, StreamOperator.not_end)

    def end(self, config: SymbolicConfiguration) -> ChannelId(1):
        self.current = None
        self.end = None
        self.transition_to(StreamOperator.start)
        return smt.Int(1)
    
    def not_end(self, config: SymbolicConfiguration, step: ChannelId(2)) -> Tuple[ChannelId(0), ChannelId(1)]:
        current = self.current
        self.current = smt.Plus(self.current, step)
        self.transition_to(StreamOperator.loop)
        return current, smt.Int(0)

    # def transition(self) -> TransitionGenerator:
    #     if self.current is None:
    #         first = yield PopChannelAction(0)
    #         end = yield PopChannelAction(1)
    #         step = yield PopChannelAction(2)

    #         self.current = smt.Plus(first, step)
    #         self.end = end
            
    #         yield PushChannelAction(0, first)
    #         yield PushChannelAction(1, smt.FALSE())

    #     else:
    #         step = yield PopChannelAction(2)

    #         def true_branch(self) -> TransitionGenerator:
    #             yield PushChannelAction(1, smt.TRUE())
    #             self.current = None
    #             self.end = None

    #         def false_branch(self) -> TransitionGenerator:
    #             yield PushChannelAction(0, self.current)
    #             yield PushChannelAction(1, smt.FALSE())
    #             self.current = smt.Plus(self.current, step)

    #         yield BranchAction(smt.GE(self.current, self.end), true_branch, false_branch)

    def copy(self) -> StreamOperator:
        copied = super().copy()
        copied.current = self.current
        copied.end = self.end
        return copied


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

    def start_3(self, config: SymbolicConfiguration, base: ChannelId(0), index: ChannelId(1), value: ChannelId(2)) -> ChannelId(0):
        config.write_memory(base, index, value)
        return smt.Int(1)

    def start_4(self, config: SymbolicConfiguration, base: ChannelId(0), index: ChannelId(1), value: ChannelId(2), sync: ChannelId(3)) -> ChannelId(0):
        config.write_memory(base, index, value)
        return smt.Int(1)


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
class SymbolicConfiguration:
    operator_states: Tuple[Operator, ...]
    channel_states: Tuple[ChannelState, ...]
    memory: List[SymbolicMemoryUpdate] = field(default_factory=list)

    path_conditions: List[smt.SMTTerm] = field(default_factory=list)
    permission_constraints: List[permission.Formula] = field(default_factory=list)

    def copy(self) -> SymbolicConfiguration:
        operator_states = tuple(operator.copy() for operator in self.operator_states)
        channel_states = tuple(state.copy() for state in self.channel_states)
        memory = list(self.memory)
        path_conditions = list(self.path_conditions)
        permission_constraints = list(self.permission_constraints)

        return SymbolicConfiguration(
            operator_states,
            channel_states,
            memory,
            path_conditions,
            permission_constraints,
        )
    
    def write_memory(self, base: smt.SMTTerm, index: smt.SMTTerm, value: smt.SMTTerm):
        self.memory.append(SymbolicMemoryUpdate(base, index, value))


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

        init_config = SymbolicConfiguration(tuple(operator_impl), tuple(channel_states))

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
        
    def get_transition_input_channels(self, pe_info: ProcessingElement, transition: TransitionFunction) -> Tuple[int, ...]:
        signature = inspect.signature(transition, eval_str=True)
        channel_ids = []
        
        for i, (name, channel_param) in enumerate(signature.parameters.items()):
            if i == 0:
                assert name == "self", f"ill-formed first transition parameter {channel_param}"
            elif i == 1:
                assert channel_param.annotation is SymbolicConfiguration, f"ill-formed second transition parameter {channel_param}"
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

    def step(self, configuration: SymbolicConfiguration) -> Tuple[SymbolicConfiguration, ...]:
        """
        Try to proceed with each transition function until one of the following happens:
        - All transitions have been tried for at least once
        - Branching
        """

        changed = False
        
        for pe_info, operator_state in \
            zip(self.graph.vertices, configuration.operator_states):

            transition = operator_state.current_transition
            
            input_channel_ids = self.get_transition_input_channels(pe_info, transition)
            input_ready = False
            
            # Check for input channel availability
            for channel_id in input_channel_ids:
                if not configuration.channel_states[channel_id].ready():
                    break
            else:
                input_ready = True

            if not input_ready:
                continue

            # TODO: when channels are bounded, check for output channel availability here

            changed = True

            # Pop values from the input channels
            input_values = []
            for channel_id in input_channel_ids:
                input_values.append(configuration.channel_states[channel_id].pop().term)

            # Run the transition
            output_values = transition(operator_state, configuration, *input_values)

            # Process output behaviors
            output_actions = self.get_transition_output_actions(transition)

            if len(output_actions) == 0:
                output_values = ()

            elif len(output_actions) == 1 and not isinstance(output_values, tuple):
                output_values = output_values,

            assert len(output_values) == len(output_actions), f"output of transition {transition} does not match its specification"

            for value, action in zip(output_values, output_actions):
                if action is Branching:
                    assert len(output_values) == 1, "cannot combine branching with other output actions"
                    assert isinstance(value, Branching), f"output of transition {transition} does not match its specification"

                    configuration_true = configuration
                    configuration_false = configuration.copy()

                    configuration_true.operator_states[pe_info.id].current_transition = value.true_branch
                    configuration_true.path_conditions.append(value.condition)

                    configuration_false.operator_states[pe_info.id].current_transition = value.false_branch
                    configuration_false.path_conditions.append(smt.Not(value.condition))

                    if not self.check_branch_feasibility(configuration_true):
                        configuration = configuration_false
                        # since the negation is unsat, the condition itself must be valid
                        configuration.path_conditions.pop(-1)
                        continue

                    if not self.check_branch_feasibility(configuration_false):
                        configuration = configuration_true
                        configuration.path_conditions.pop(-1)
                        continue

                    return configuration_true, configuration_false
                
                elif isinstance(action, ChannelId) and action.id in pe_info.outputs:
                    channels = pe_info.outputs[action.id]
                    for channel in channels:
                        configuration.channel_states[channel.id].push(self.get_fresh_permissioned_value(value))

        if not changed:
            return ()

        return configuration,
