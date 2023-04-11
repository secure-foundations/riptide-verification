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
    
    def __init__(self, pe: ProcessingElement, permission: permission.PermissionVariable, transition: Optional[TransitionFunction] = None):
        self.pe = pe
        self.internal_permission: permission.PermissionVariable = permission
        self.current_transition: TransitionFunction = transition or type(self).start

    def transition_to(self, transition: TransitionFunction):
        self.current_transition = transition

    def start(self, config: SymbolicConfiguration):
        raise NotImplementedError()
    
    def copy(self) -> Operator:
        return type(self)(self.pe, self.internal_permission, self.current_transition)


@Operator.implement("ARITH_CFG_OP_ADD")
class AddOperator(Operator):
    def start(self, config: SymbolicConfiguration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        # print("plus", a, b)
        return smt.Plus(a, b)
    

@Operator.implement("MUL_CFG_OP_MUL")
class AddOperator(Operator):
    def start(self, config: SymbolicConfiguration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        # print("mul", a, b)
        return smt.Times(a, b)
    

@Operator.implement("ARITH_CFG_OP_SGT")
class SignedGreaterThanOperator(Operator):
    def start(self, config: SymbolicConfiguration, a: ChannelId(0), b: ChannelId(1)) -> ChannelId(0):
        return smt.Ite(smt.GT(a, b), smt.Int(1), smt.Int(0))


@Operator.implement("CF_CFG_OP_SELECT")
class SelectOperator(Operator):
    def start(self, config: SymbolicConfiguration, decider: ChannelId(0)) -> Branching:
        return Branching(smt.Equals(decider, smt.Int(0)), SelectOperator.false, SelectOperator.true)
    
    def true(self, config: SymbolicConfiguration, a: ChannelId(1), b: ChannelId(2)) -> ChannelId(0):
        return a

    def false(self, config: SymbolicConfiguration, a: ChannelId(1), b: ChannelId(2)) -> ChannelId(0):
        return b


@Operator.implement("CF_CFG_OP_CARRY")
class CarryOperator(Operator):
    def start(self, config: SymbolicConfiguration, a: ChannelId(1)) -> ChannelId(0):
        self.transition_to(CarryOperator.loop)
        return a
    
    def loop(self, config: SymbolicConfiguration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.Int(0)), CarryOperator.pass_b, CarryOperator.start)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.Int(0)), CarryOperator.start, CarryOperator.pass_b)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def pass_b(self, config: SymbolicConfiguration, b: ChannelId(2)) -> ChannelId(0):
        self.transition_to(CarryOperator.loop)
        return b


@Operator.implement("CF_CFG_OP_STEER")
class SteerOperator(Operator):
    def start(self, config: SymbolicConfiguration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.Int(0)), SteerOperator.pass_value, SteerOperator.discard_value)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.Int(0)), SteerOperator.discard_value, SteerOperator.pass_value)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def pass_value(self, config: SymbolicConfiguration, value: ChannelId(1)) -> ChannelId(0):
        self.transition_to(SteerOperator.start)
        return value
    
    def discard_value(self, config: SymbolicConfiguration, value: ChannelId(1)):
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

    def start(self, config: SymbolicConfiguration, value: ChannelId(1)) -> ChannelId(0):
        self.transition_to(InvariantOperator.loop)
        self.value = value
        return value
    
    def loop(self, config: SymbolicConfiguration, decider: ChannelId(0)) -> Branching:
        if self.pe.pred == "CF_CFG_PRED_FALSE":
            return Branching(smt.Equals(decider, smt.Int(0)), InvariantOperator.invariant, InvariantOperator.start)
        elif self.pe.pred == "CF_CFG_PRED_TRUE":
            return Branching(smt.Equals(decider, smt.Int(0)), InvariantOperator.start, InvariantOperator.invariant)
        else:
            assert False, f"unknown pred {self.pe.pred}"

    def invariant(self, config: SymbolicConfiguration) -> ChannelId(0):
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

    def start(self, config: SymbolicConfiguration, first: ChannelId(0), end: ChannelId(1)):
        self.current = first
        self.end = end
        # print("start", str(self.pe.id), self.current, self.end)
        self.transition_to(StreamOperator.loop)
        return ()

    def loop(self, config: SymbolicConfiguration) -> Branching:
        # print("loop", str(self.pe.id), self.current, self.end)
        return Branching(smt.GE(self.current, self.end), StreamOperator.done, StreamOperator.not_done)

    def done(self, config: SymbolicConfiguration) -> ChannelId(1):
        self.current = None
        self.end = None
        self.transition_to(StreamOperator.start)

        # print("end", str(self.pe.id))

        if self.pe.pred == "STREAM_CFG_PRED_FALSE":
            return smt.Int(1)
        elif self.pe.pred == "STREAM_CFG_PRED_TRUE":
            return smt.Int(0)
        else:
            assert False, f"unknown pred {self.pe.pred}"
    
    def not_done(self, config: SymbolicConfiguration, step: ChannelId(2)) -> Tuple[ChannelId(0), ChannelId(1)]:
        current = self.current
        self.current = smt.Plus(self.current, step)
        self.transition_to(StreamOperator.loop)

        # print("not_end", str(self.pe.id))

        if self.pe.pred == "STREAM_CFG_PRED_FALSE":
            done_flag = smt.Int(0)
        elif self.pe.pred == "STREAM_CFG_PRED_TRUE":
            done_flag = smt.Int(1)
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

    def start_3(self, config: SymbolicConfiguration, base: ChannelId(0), index: ChannelId(1), value: ChannelId(2)) -> ChannelId(0):
        config.write_memory(base, index, value)
        return smt.Int(1)

    def start_4(self, config: SymbolicConfiguration, base: ChannelId(0), index: ChannelId(1), value: ChannelId(2), sync: ChannelId(3)) -> ChannelId(0):
        config.write_memory(base, index, value)
        return smt.Int(1)
    

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

    def start_2(self, config: SymbolicConfiguration, base: ChannelId(0), index: ChannelId(1)) -> ChannelId(0):
        return config.read_memory(base, index)

    def start_3(self, config: SymbolicConfiguration, base: ChannelId(0), index: ChannelId(1), sync: ChannelId(2)) -> ChannelId(0):
        return config.read_memory(base, index)


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

    # Memory is currently modelled as a map
    # base |-> array of ints
    # This is assuming all bases represent disjoint regions of memory
    memory: List[SymbolicMemoryUpdate] = field(default_factory=list)
    memory_var: smt.SMTTerm = field(default_factory=lambda: SymbolicConfiguration.get_fresh_memory_var())
    memory_constraints: List[smt.SMTTerm] = field(default_factory=list)

    path_constraints: List[smt.SMTTerm] = field(default_factory=list)
    permission_constraints: List[permission.Formula] = field(default_factory=list)

    @staticmethod
    def get_fresh_memory_var() -> smt.SMTTerm:
        return smt.FreshSymbol(smt.ArrayType(smt.INT, smt.ArrayType(smt.INT, smt.INT)))

    def copy(self) -> SymbolicConfiguration:
        return SymbolicConfiguration(
            tuple(operator.copy() for operator in self.operator_states),
            tuple(state.copy() for state in self.channel_states),
            list(self.memory),
            self.memory_var,
            list(self.memory_constraints),
            list(self.path_constraints),
            list(self.permission_constraints),
        )
    
    def write_memory(self, base: smt.SMTTerm, index: smt.SMTTerm, value: smt.SMTTerm):
        self.memory.append(SymbolicMemoryUpdate(base, index, value))

        new_var = SymbolicConfiguration.get_fresh_memory_var()
        self.memory_constraints.append(smt.Equals(
            new_var,
            smt.Store(self.memory_var, base, smt.Store(smt.Select(self.memory_var, base), index, value)),
        ))
        self.memory_var = new_var

    def read_memory(self, base: smt.SMTTerm, index: smt.SMTTerm) -> smt.SMTTerm:
        return smt.Select(smt.Select(self.memory_var, base), index)


class SymbolicExecutor:
    def __init__(self, graph: DataflowGraph, free_vars: Mapping[str, smt.SMTTerm]):
        self.graph = graph

        self.free_vars = dict(free_vars)
        self.permission_var_count = 0
        
        initial_permissions: List[permission.PermissionVariable] = []

        # Initialize operator implementations
        operator_impl: List[Operator] = []
        for i, vertex in enumerate(graph.vertices):
            assert vertex.operator in Operator.OPERATOR_IMPL_MAP, \
                   f"unable to find an implementation for operator {vertex.operator}"
            impl = Operator.OPERATOR_IMPL_MAP[vertex.operator]
            perm_var = self.get_fresh_permission_var(f"internal-{i}-")
            initial_permissions.append(perm_var)
            operator_impl.append(impl(vertex, perm_var))

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

                initial_permissions.append(value.permission)

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

        # All initial permissions have to be disjoint
        init_config.permission_constraints.append(permission.Disjoint(tuple(initial_permissions)))

        self.configurations: List[SymbolicConfiguration] = [init_config]

    def get_fresh_permissioned_value(self, value: smt.SMTTerm) -> PermissionedValue:
        return PermissionedValue(value, self.get_fresh_permission_var())

    def get_fresh_permission_var(self, prefix="p") -> permission.PermissionVariable:
        var = permission.PermissionVariable(f"{prefix}{self.permission_var_count}")
        self.permission_var_count += 1
        return var
    
    def check_branch_feasibility(self, configuration: SymbolicConfiguration) -> bool:
        with smt.Solver(name="z3") as solver:
            for path_condition in configuration.path_constraints:
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
    
        for pe_info in self.graph.vertices:
            operator_state = configuration.operator_states[pe_info.id]

            transition = operator_state.current_transition
            
            input_channel_ids = self.get_transition_input_channels(pe_info, transition)
            input_ready = True
            
            # Check for input channel availability
            # print(f"{pe_info.id} {input_channel_ids}")
            for channel_id in input_channel_ids:
                if not configuration.channel_states[channel_id].ready():
                    # print(type(operator_state), f"not ready: {input_channel_ids}")
                    input_ready = False
                    break

            if not input_ready:
                continue

            # print("triggering", str(pe_info.id))

            # TODO: when channels are bounded, check for output channel availability here

            changed = True

            # Pop values from the input channels
            input_permissions: List[permission.PermissionVariable] = [operator_state.internal_permission]
            input_values = []
            for channel_id in input_channel_ids:
                value = configuration.channel_states[channel_id].pop()
                input_permissions.append(value.permission)
                input_values.append(value.term)

            # Run the transition
            # print(f"original transition {operator_state.current_transition}")
            output_values = transition(operator_state, configuration, *input_values)
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

                mem_permission = permission.ReadPermission(var_name) if isinstance(operator_state, LoadOperator) else \
                                 permission.WritePermission(var_name)
                
                configuration.permission_constraints.append(permission.Inclusion(mem_permission, permission.DisjointUnion(input_permissions)))

            # Update internal permission
            operator_state.internal_permission = self.get_fresh_permission_var(f"internal-{pe_info.id}-")
            output_permissions: List[permission.PermissionVariable] = [operator_state.internal_permission]

            if len(output_actions) == 1 and output_actions[0] is Branching:
                # A single branching action
                output_value = output_values[0]
                assert isinstance(output_value, Branching), f"output of transition {transition} does not match its specification"

                configuration_true = configuration
                configuration_false = configuration.copy()

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

                if not self.check_branch_feasibility(configuration_true):
                    configuration = configuration_false
                    # since the negation is unsat, the condition itself must be valid
                    configuration.path_constraints.pop(-1)
                    continue

                if not self.check_branch_feasibility(configuration_false):
                    configuration = configuration_true
                    configuration.path_constraints.pop(-1)
                    continue

                return configuration_true, configuration_false
            else:
                # Output channels
                for value, action in zip(output_values, output_actions):
                    assert isinstance(action, ChannelId), f"unexpected action {action}"
                    if action.id not in pe_info.outputs:
                        continue

                    channels = pe_info.outputs[action.id]
                    for channel in channels:
                        permissioned_value = self.get_fresh_permissioned_value(value.simplify())
                        configuration.channel_states[channel.id].push(permissioned_value)
                        output_permissions.append(permissioned_value.permission)

                # Add permission constraint:
                #   output_permission + new_internal_permission <= input_permission + old_internal_permission
                permission_constraint = permission.Inclusion(
                    permission.DisjointUnion.of(*output_permissions),
                    permission.DisjointUnion.of(*input_permissions),
                )

                configuration.permission_constraints.append(permission_constraint)

        if not changed:
            return ()

        return configuration,
