from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from collections import OrderedDict

import json
import time
import argparse
import itertools

from semantics.dataflow import *


@dataclass(frozen=True)
class ConfigurationShape:
    graph: DataflowGraph
    operator_transitions: Tuple[TransitionFunction, ...]
    channel_size: Tuple[int, ...]

    @staticmethod
    def from_config(config: Configuration) -> ConfigurationShape:
        return ConfigurationShape(
            config.graph,
            tuple(op_state.current_transition for op_state in config.operator_states),
            tuple(channel_state.count() if not channel_state.hold_constant else 0 for channel_state in config.channel_states),
        )

    def __eq__(self, other) -> bool:
        return (self.operator_transitions == other.operator_transitions and
                self.channel_size == other.channel_size)

    def __hash__(self) -> int:
        return hash(self.operator_transitions) ^ hash(self.channel_size)


@dataclass
class ShapePartition:
    """
    Maintains a partition of configurations by their shape
    """

    shape_map: OrderedDict[ConfigurationShape, List[Configuration]] = field(default_factory=OrderedDict)

    def __contains__(self, shape: ConfigurationShape) -> bool:
        return shape in self.shape_map

    def add(self, config: Configuration):
        shape = ConfigurationShape.from_config(config)
        if shape not in self.shape_map:
            self.shape_map[shape] = []
        self.shape_map[shape].append(config)

    def add_all(self, configs: Iterable[Configuration]):
        for config in configs:
            self.add(config)

    def __getitem__(self, key: ConfigurationShape) -> Iterable[Configuration]:
        return self.shape_map[key]

    def __len__(self) -> int:
        return len(self.shape_map)

    def shapes(self) -> Iterable[ConfigurationShape]:
        return self.shape_map.keys()


@dataclass
class ShapeGraph:
    """
    A directed graph storing the transition relation between shapes
    """

    shapes: List[ConfigurationShape] = field(default_factory=list)
    shape_to_index: Dict[ConfigurationShape, int] = field(default_factory=dict)

    incoming_edges: Dict[int, Set[int]] = field(default_factory=dict)
    outgoing_edges: Dict[int, Set[int]] = field(default_factory=dict)

    def add_shape(self, shape: ConfigurationShape):
        if shape not in self.shape_to_index:
            self.shape_to_index[shape] = len(self.shapes)
            self.shapes.append(shape)

    def get_shapes(self) -> Iterable[ConfigurationShape]:
        return self.shapes

    def get_incoming_shapes(self, shape: ConfigurationShape) -> Tuple[ConfigurationShape, ...]:
        shape_index = self.shape_to_index[shape]
        indices = sorted(self.incoming_edges.get(shape_index, set()))
        return tuple(self.shapes[index] for index in indices)

    def get_outgoing_shapes(self, shape: ConfigurationShape) -> Tuple[ConfigurationShape, ...]:
        shape_index = self.shape_to_index[shape]
        indices = sorted(self.outgoing_edges.get(shape_index, set()))
        return tuple(self.shapes[index] for index in indices)

    def add_edge(self, shape1: ConfigurationShape, shape2: ConfigurationShape):
        shape_index1 = self.shape_to_index[shape1]
        shape_index2 = self.shape_to_index[shape2]

        if shape_index2 not in self.incoming_edges:
            self.incoming_edges[shape_index2] = set()

        if shape_index1 not in self.outgoing_edges:
            self.outgoing_edges[shape_index1] = set()

        # assert shape_index1 not in self.incoming_edges[shape_index2]
        # assert shape_index2 not in self.outgoing_edges[shape_index1]

        self.incoming_edges[shape_index2].add(shape_index1)
        self.outgoing_edges[shape_index1].add(shape_index2)


@dataclass
class FirebilityDependencyGraph:
    """
    Fix a configuration (shape) C

    Firebility dependency graph is a directed graph where nodes are PEs
    And a -> b iff a in C is fireable only after b is fired

    If a -> None, then a is not fireable any more

    Dependencies are separated into two sets:
    Input dependency: a i-> b iff a is waiting on b's output
    Output dependency: a o->b iff b is waiting on a's execution, which will empty a slot in a's output channel
    """

    graph: DataflowGraph

    input_dependency: Dict[int, Set[Optional[int]]] = field(default_factory=dict)
    output_dependency: Dict[int, Set[Optional[int]]] = field(default_factory=dict)

    def add_input_dependency(self, pe_id1: int, pe_id2: Optional[int]):
        if pe_id1 not in self.input_dependency:
            self.input_dependency[pe_id1] = set()
        self.input_dependency[pe_id1].add(pe_id2)

    def add_output_dependency(self, pe_id1: int, pe_id2: Optional[int]):
        if pe_id1 not in self.output_dependency:
            self.output_dependency[pe_id1] = set()
        self.output_dependency[pe_id1].add(pe_id2)

    @staticmethod
    def from_shape(shape: ConfigurationShape) -> FirebilityDependencyGraph:
        graph = FirebilityDependencyGraph(shape.graph)

        for pe, transition in zip(shape.graph.vertices, shape.operator_transitions):
            input_channel_ids = Configuration.get_transition_input_channels(pe, transition)
            output_channel_ids = Configuration.get_transition_output_channels(pe, transition)

            for channel_id in input_channel_ids:
                channel = shape.graph.channels[channel_id]

                if channel.hold:
                    continue

                if shape.channel_size[channel_id] == 0:
                    # In particular, if channel.source is None,
                    # pe.id is not fireable any more (i.e. pe -> None)
                    graph.add_input_dependency(pe.id, channel.source)

            for channel_id in output_channel_ids:
                channel = shape.graph.channels[channel_id]

                assert not channel.hold

                if channel.bound and shape.channel_size[channel_id] >= channel.bound:
                    # In particular, if channel.destination is None,
                    # pe.id is not fireable any more (i.e. pe -> None)
                    graph.add_output_dependency(pe.id, channel.destination)

        return graph

    def find_cycle_with_output_dependency(self) -> Tuple[int, ...]:
        """
        Find one cycle in the graph with output dependency:
        that is, there is some unused value in the cycle, yet
        none of the PEs involved is fireable any more
        """

        visited = set()
        current_path = []

        def visit(pe_id, output_dep_only=False) -> bool:
            visited.add(pe_id)
            current_path.append(pe_id)

            neighbors = self.output_dependency.get(pe_id, set())

            if not output_dep_only:
                neighbors = neighbors.union(self.input_dependency.get(pe_id, set()))

            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                if visit(neighbor):
                    return True

                if neighbor in current_path:
                    return True

            current_path.pop()
            return False

        # Start with a node with output dependency
        for pe_id in self.output_dependency.keys():
            if pe_id in visited:
                continue

            assert len(current_path) == 0

            if visit(pe_id, output_dep_only=True):
                return tuple(current_path)

        return ()


"""
Schedule is a function that picks a list of fireable
operators in any configuration to explore (in parallel)
"""
OperatorSchedule = Callable[[Configuration], Tuple[int, ...]]


def deterministic_schedule(config: Configuration) -> Tuple[int, ...]:
    fireable_operators = [ pe for pe in config.graph.vertices if config.is_fireable(pe.id) ]
    if len(fireable_operators) != 0:
        fireable_operators.sort(key=lambda pe: pe.id)
        return fireable_operators[0].id,
    return ()


def pure_priority_schedule(config: Configuration) -> Tuple[int, ...]:
    """
    Let F be the set of fireable PEs

    If F has any pure operator in it, this schedule selects the
    first pure operator based on the PE id index.

    Otherwise we return the entire F:
    - If F is empty, this means we terminate
    - If F is not empty, then F must all be memory operators, then we
    need to explore an arbitrary interleaving of these operators
    """

    fireable_operators = [ pe for pe in config.graph.vertices if config.is_fireable(pe.id) ]

    fireable_pure_operators = [
        pe
        for pe in fireable_operators
        if pe.operator != "MEM_CFG_OP_STORE" and
        pe.operator != "MEM_CFG_OP_LOAD"
    ]

    if len(fireable_pure_operators) != 0:
        fireable_pure_operators.sort(key=lambda pe: pe.id)
        return fireable_pure_operators[0].id,

    return tuple(op.id for op in fireable_operators)


def arbitrary_interleaving_schedule(config: Configuration) -> Tuple[int, ...]:
    fireable_operators = [ pe for pe in config.graph.vertices if config.is_fireable(pe.id) ]
    fireable_operators.sort(key=lambda pe: pe.id)
    return tuple(op.id for op in fireable_operators)


def explore_states(
    initial: Configuration,
    schedule: OperatorSchedule,
    max_depth: int,
) -> Tuple[Tuple[Configuration], ShapePartition]:
    """
    Explore the tree of symbolic states from an initial configuration,
    up to a certain depth
    """

    # [(config, depth)]
    queue: List[Tuple[Configuration, int]] = [(initial, 0)]
    explored_states: List[Configuration] = []

    max_branching: int = 1

    partitions = ShapePartition()
    shape_graph = ShapeGraph()

    symbolic_execution_time = 0.0
    start_time = time.process_time()

    try:
        # BFS on the symbolic configurations
        while len(queue) != 0:
            config, depth = queue.pop(0)
            explored_states.append(config)

            shape = ConfigurationShape.from_config(config)
            shape_graph.add_shape(shape)

            partitions.add(config)

            if len(explored_states) % 100 == 0:
                num_redundant = 0
                num_final = 0
                num_initial = 0
                num_branching = 0
                num_merging = 0
                num_other = 0

                other_fan_ins = []
                other_fan_outs = []

                for explored_shape in shape_graph.get_shapes():
                    if (len(shape_graph.get_incoming_shapes(explored_shape)) == 1 and
                        len(shape_graph.get_outgoing_shapes(explored_shape)) == 1):
                        num_redundant += 1

                    elif len(shape_graph.get_outgoing_shapes(explored_shape)) == 0:
                        num_final += 1

                    elif len(shape_graph.get_incoming_shapes(explored_shape)) == 0:
                        num_initial += 1

                    elif (len(shape_graph.get_incoming_shapes(explored_shape)) == 1 and
                        len(shape_graph.get_outgoing_shapes(explored_shape)) > 1):
                        num_branching += 1

                    elif (len(shape_graph.get_incoming_shapes(explored_shape)) > 1 and
                        len(shape_graph.get_outgoing_shapes(explored_shape)) == 1):
                        num_merging += 1

                    else:
                        num_other += 1
                        other_fan_ins.append(len(shape_graph.get_incoming_shapes(explored_shape)))
                        other_fan_outs.append(len(shape_graph.get_outgoing_shapes(explored_shape)))

                print(
                    f"{len(explored_states)} state(s) explored, depth {depth}, "
                    f"{len(partitions)} partition(s) found ({num_redundant} redundant, {num_final} final, {num_initial} initial, {num_branching} branching, {num_merging} merging, {num_other} other), "
                    f"symbolic execution time {round(symbolic_execution_time, 2)} s, "
                    f"total time {round(time.process_time() - start_time, 2)} s"
                )
                print(*zip(other_fan_ins, other_fan_outs))

            if depth >= max_depth:
                continue

            operators_to_fire = schedule(config)

            num_branches = 0

            for pe_id in operators_to_fire:
                start = time.process_time()
                results = config.copy().step_exhaust(pe_id)
                symbolic_execution_time += time.process_time() - start

                for result in results:
                    assert isinstance(result, NextConfiguration)

                    result_shape = ConfigurationShape.from_config(result.config)
                    if result_shape not in partitions:
                        shape_graph.add_shape(result_shape)
                    shape_graph.add_edge(shape, result_shape)

                    queue.append((result.config, depth + 1))
                    num_branches += 1

            if num_branches > max_branching:
                max_branching = num_branches

    except:
        print("final state (examples):")
        # for explored_shape in shape_graph.get_shapes():
        #     if len(shape_graph.get_outgoing_shapes(explored_shape)) == 0 and explored_shape in partitions:
        #         print(partitions.shape_map[explored_shape][0])

    print("max branching", max_branching)

    return tuple(explored_states), partitions


def generalize_partition(partition: Tuple[Configuration, ...]) -> Configuration:
    """
    Produce a symbolic configuration that is more general than all of the
    symbolic configurations in the partition
    """

    dummy_perm = permission.Variable("dummy")

    cut_point = partition[0].copy()

    # Memory updates are abstracted away
    # TODO: we may be able to learn some invariants about the memory too?
    cut_point.memory_updates = []
    cut_point.memory = Configuration.get_fresh_memory_var()

    # Clear path conditions
    cut_point.path_conditions = []

    # Map each cut point value to a list of corresponding terms in the configs in the partition
    value_correspondence: OrderedDict[smt.SMTTerm, Tuple[smt.SMTTerm, ...]] = OrderedDict()

    for pe in cut_point.graph.vertices:
        # The transition is already set, we now need to infer internal state values
        # if they are required (e.g. for Invariant)

        if pe.operator == "CF_CFG_OP_INVARIANT":
            assert isinstance(cut_point.operator_states[pe.id], InvariantOperator)

            if cut_point.operator_states[pe.id].value is not None:
                value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_inv_{pe.id}_%d")
                value_correspondence[value] = tuple(config.operator_states[pe.id].value for config in partition)
                cut_point.operator_states[pe.id].value = value

        elif pe.operator == "STREAM_FU_CFG_T":
            assert isinstance(cut_point.operator_states[pe.id], StreamOperator)

            if cut_point.operator_states[pe.id].current is not None:
                current_value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_stream_current_{pe.id}_%d")
                end_value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_stream_end_{pe.id}_%d")
                value_correspondence[current_value] = tuple(config.operator_states[pe.id].current for config in partition)
                value_correspondence[end_value] = tuple(config.operator_states[pe.id].end for config in partition)
                cut_point.operator_states[pe.id].current = current_value
                cut_point.operator_states[pe.id].end = end_value

    for channel in cut_point.graph.channels:
        if channel.hold or channel.constant is not None:
            continue

        for index in range(cut_point.channel_states[channel.id].count()):
            value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_channel_{channel.id}_{index}_%d")
            value_correspondence[value] = tuple(config.channel_states[channel.id].values[index].term for config in partition)
            cut_point.channel_states[channel.id].values[index] = PermissionedValue(value, dummy_perm)

    # Add some basic equality constraints:
    # output channels of the same port must have the same prefixes
    for pe in cut_point.graph.vertices:
        for same_port_output_channels in pe.outputs.values():
            for i, channel1 in enumerate(same_port_output_channels):
                for channel2 in same_port_output_channels[i + 1:]:
                    for value1, value2 in zip(cut_point.channel_states[channel1.id].values[::-1], cut_point.channel_states[channel2.id].values[::-1]):
                        # print(smt.Equals(value1.term, value2.term))
                        cut_point.path_conditions.append(smt.Equals(value1.term, value2.term))

    # Learn some potential equality constraints
    # for i, channel1 in enumerate(cut_point.graph.channels):
    #     for channel2 in cut_point.graph.channels[i + 1:]:
    #         for j, value1 in enumerate(cut_point.channel_states[channel1.id].values):
    #             for k, value2 in enumerate(cut_point.channel_states[channel2.id].values):
    #                 for config in partition:
    #                     if config.channel_states[channel1.id].values[j].term != config.channel_states[channel2.id].values[k].term:
    #                         break
    #                 else:
    #                     # All values in these two sets have the same value (syntactically)
    #                     # so infer an equality between them
    #                     cut_point.path_conditions.append(smt.Equals(value1.term, value2.term))

    all_path_conditions: Tuple[smt.SMTTerm, ...] = tuple(smt.And(config.path_conditions) for config in partition)

    unary_predicates = [
        lambda t: smt.Equals(t, smt.BVConst(0, WORD_WIDTH)),
    ]

    # Learn unary predicates from the values
    for cut_point_value, partition_values in value_correspondence.items():
        for predicate in unary_predicates:
            with smt.push_solver(cut_point.solver):
                # Check if the predicate holds for all values in partition configs
                cut_point.solver.add_assertion(smt.Not(smt.And(
                    smt.Implies(path_condition, predicate(partition_value))
                    for path_condition, partition_value in zip(all_path_conditions, partition_values, strict=True)
                )))

                if not cut_point.solver.solve():
                    # The predicate is valid
                    cut_point.path_conditions.append(predicate(cut_point_value))
                    continue

            # Also check the negation
            with smt.push_solver(cut_point.solver):
                # Check if the predicate holds for all values in partition configs
                cut_point.solver.add_assertion(smt.Not(smt.And(
                    smt.Implies(path_condition, smt.Not(predicate(partition_value)))
                    for path_condition, partition_value in zip(all_path_conditions, partition_values, strict=True)
                )))

                if not cut_point.solver.solve():
                    # The predicate is valid
                    cut_point.path_conditions.append(smt.Not(predicate(cut_point_value)))

    # Learn binary predicates
    # NOTE: this seems to make things a lot slower without much help in reducing the state space
    # binary_predicates = [
    #     lambda t, s: smt.BVSLT(t, s),
    # ]
    # for (cut_point_value1, partition_values1), (cut_point_value2, partition_values2) in itertools.permutations(value_correspondence.items(), 2):
    #     for predicate in binary_predicates:
    #         with smt.push_solver(cut_point.solver):
    #             # Check if the predicate holds for all values in partition configs
    #             cut_point.solver.add_assertion(smt.Not(smt.And(
    #                 smt.Implies(path_condition, predicate(partition_value1, partition_value2))
    #                 for path_condition, partition_value1, partition_value2 in zip(all_path_conditions, partition_values1, partition_values2, strict=True)
    #             )))

    #             if not cut_point.solver.solve():
    #                 # The predicate is valid
    #                 # print("learned binary", predicate(cut_point_value1, cut_point_value2))
    #                 cut_point.path_conditions.append(predicate(cut_point_value1, cut_point_value2))
    #                 continue

    #         # Also check the negation
    #         with smt.push_solver(cut_point.solver):
    #             # Check if the predicate holds for all values in partition configs
    #             cut_point.solver.add_assertion(smt.Not(smt.And(
    #                 smt.Implies(path_condition, smt.Not(predicate(partition_value1, partition_value2)))
    #                 for path_condition, partition_value1, partition_value2 in zip(all_path_conditions, partition_values1, partition_values2, strict=True)
    #             )))

    #             if not cut_point.solver.solve():
    #                 # The predicate is valid
    #                 # print("learned binary", smt.Not(predicate(cut_point_value1, cut_point_value2)))
    #                 cut_point.path_conditions.append(smt.Not(predicate(cut_point_value1, cut_point_value2)))
    #                 continue

    # for config in partition:
    #     match_result, _ = cut_point.match(config)
    #     assert isinstance(match_result, MatchingSuccess)

    #     if not match_result.check_condition():
    #         print(cut_point, config)
    #         print(match_result.condition)
    #         exit()

    return cut_point


def check_cut_point_abstraction_deadlock_freedom(
    cut_points: Tuple[Configuration, ...],
    schedule: OperatorSchedule,
    solver: smt.SMTSolver,
):
    """
    Check if the cut point abstraction is valid and deadlock-free
    """

    reachable_shapes: Set[ConfigurationShape] = set()

    shape_to_cut_point = { ConfigurationShape.from_config(cut_point): cut_point for cut_point in cut_points }
    num_final_cut_points = 0

    for cut_point in cut_points:
        queue = [cut_point]

        while len(queue) != 0:
            config = queue.pop(0)

            shape = ConfigurationShape.from_config(config)
            reachable_shapes.add(shape)

            if len(check_deadlock(shape)) != 0:
                print("found deadlock at a potentially reachable config")
                print(config)
                assert False

            if config.is_final():
                if config == cut_point:
                    num_final_cut_points += 1
                    break

                print("found unmatched final state")
                print(config)
                assert False

            for pe_id in schedule(config):
                results = config.copy().step_exhaust(pe_id)

                # match against another cut points
                for result in results:
                    assert isinstance(result, NextConfiguration)

                    # Check if matched by some cut point
                    result_shape = ConfigurationShape.from_config(result.config)
                    if result_shape in shape_to_cut_point:
                        other_cut_point = shape_to_cut_point[result_shape]
                        match_result, _ = other_cut_point.match(result.config)
                        if isinstance(match_result, MatchingSuccess) and match_result.check_condition(solver):
                            continue

                    queue.append(result.config)

    print(f"{num_final_cut_points} terminating cut point(s)")
    print(f"{len(reachable_shapes)} reachable shape(s) in the cut point abstraction")


def construct_cut_point_abstraction(
    configs: Iterable[Configuration],
    schedule: OperatorSchedule,
    solver: smt.SMTSolver,
    unmatch_explore_depth: int = 15,
) -> Tuple[Configuration, ...]:
    """
    Check if the given set of cut points is a valid abstraction of the concrete semantics
    (that is they are closed under execution)

    If so, return the same set
    Otherwise, we may append new cut points and generate a final version
    """

    partitions = ShapePartition()
    partitions.add_all(configs)

    # Construct a cut point for each partition
    cut_points: OrderedDict[ConfigurationShape, Configuration] = \
        OrderedDict((shape, generalize_partition(partitions[shape])) for shape in partitions.shapes())

    shapes_to_check: List[ConfigurationShape] = list(partitions.shapes())

    while len(shapes_to_check) != 0:
        unmatched_configs: List[Configuration] = []

        for shape in shapes_to_check:
            # print(f"checking cut point {i}")

            if cut_points[shape].is_final():
                continue

            queue: List[Tuple[Configuration, int]] = [(cut_points[shape], 0)]

            # explored_partition = ShapePartition()
            # explored_shape_graph = ShapeGraph()
            # explored_shape_graph.add_shape(shape)
            # explored_partition.add(cut_points[shape])

            while len(queue) != 0:
                config, depth = queue.pop(0)

                if depth >= unmatch_explore_depth:
                    unmatched_configs.append(config)
                    continue

                pe_to_fire = schedule(config)

                if len(pe_to_fire) == 0:
                    unmatched_configs.append(config)

                for pe_id in pe_to_fire:
                    results = config.copy().step_exhaust(pe_id)

                    # match against another cut points
                    for result in results:
                        assert isinstance(result, NextConfiguration)

                        result_shape = ConfigurationShape.from_config(result.config)

                        # explored_partition.add(result.config)
                        # explored_shape_graph.add_shape(result_shape)
                        # explored_shape_graph.add_edge(ConfigurationShape.from_config(config), result_shape)

                        if result_shape in partitions:
                            other_cut_point = cut_points[result_shape]
                            match_result, _ = other_cut_point.match(result.config)
                            if isinstance(match_result, MatchingSuccess):
                                if not match_result.check_condition(solver):
                                    unmatched_configs.append(result.config)
                            else:
                                assert False, "same shape without match"
                        else:
                            # New shape
                            queue.append((result.config, depth + 1))

        # for new_shape in explored_shape_graph.get_shapes():
        #     # Only add configs in a shape if it
        #     # * Has more than 1 incoming edge, or
        #     # * Has no successor

        #     in_deg = len(explored_shape_graph.get_incoming_shapes(new_shape))
        #     out_deg = len(explored_shape_graph.get_outgoing_shapes(new_shape))

        #     if in_deg > 1 or (in_deg == 1 and new_shape == shape) or out_deg == 0:
        #         if new_shape in explored_partition:
        #             unmatched_configs.extend(explored_partition[new_shape])

        shapes_to_check = []

        # Still found configurations that are not matched
        # Refine partition and cut points again
        if len(unmatched_configs) != 0:
            num_new = 0
            # num_skipped = 0

            updated_shapes: OrderedDict[ConfigurationShape, None] = OrderedDict()

            for config in unmatched_configs:
                shape = ConfigurationShape.from_config(config)
                if shape not in partitions:
                    # Try to run the config further to see if it ends up in an existing shape
                    # cycle = check_deadlock(shape)
                    # if len(cycle) != 0:
                    #     print(f"found cycle {cycle} in config")
                    #     print(partitions[shape][0])
                    #     assert False
                    num_new += 1

                updated_shapes[shape] = None
                partitions.add(config)

            shapes_to_check = list(updated_shapes.keys())

            for shape in shapes_to_check:
                cut_points[shape] = generalize_partition(partitions[shape])

            # for shape in updated_shapes:
            #     new_cut_point = generalize_partition(partitions[shape])

            #     if shape in cut_points:
            #         # If the cut point has not become more general, no need to recheck
            #         match_result, _ = cut_points[shape].match(new_cut_point)
            #         if isinstance(match_result, MatchingSuccess) and match_result.check_condition():
            #             num_skipped += 1
            #             continue

            #     cut_points[shape] = new_cut_point
            #     shapes_to_check.append(shape)

            print(
                f"found {len(unmatched_configs)} unmatched config(s), "
                f"{num_new} new partition(s), "
                f"{len(updated_shapes) - num_new} updated partition(s), "
                f"total {len(partitions)} partition(s)"
            )

    return tuple(cut_points.values())


def check_deadlock(shape: ConfigurationShape) -> Tuple[int, ...]:
    """
    Return which operators are in a deadlock
    """
    return FirebilityDependencyGraph.from_shape(shape).find_cycle_with_output_dependency()


function_arg_free_vars: Set[smt.SMTTerm] = set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dfg", help="Input dataflow graph")
    parser.add_argument("--channel-bound", type=int, default=1, help="Channel buffer size")
    args = parser.parse_args()

    with open(args.dfg) as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source), args.channel_bound)

    free_vars: Dict[str, smt.SMTTerm] = {}

    for function_arg in dfg.function_arguments:
        if function_arg.variable_name not in free_vars:
            free_vars[function_arg.variable_name] = smt.FreshSymbol(smt.BVType(WORD_WIDTH))
            function_arg_free_vars.add(free_vars[function_arg.variable_name])

    # with smt.Solver("cvc5", logic="QF_AUFBV") as solver:
    with smt.Solver("z3", logic="QF_AUFBV", random_seed=0) as solver:
        initial = Configuration.get_initial_configuration(dfg, free_vars, disable_permissions=True, solver=solver)

        # _, partitions = explore_states(initial, pure_priority_schedule, 1000)
        # for part in partitions.shape_map.values():
        #     print(part[0])

        start_time = time.process_time()
        cut_points = construct_cut_point_abstraction([initial], pure_priority_schedule, solver)

        print(f"found {len(cut_points)} cut points in {round(time.process_time() - start_time, 2)} s")

        start_time = time.process_time()
        check_cut_point_abstraction_deadlock_freedom(cut_points, pure_priority_schedule, solver)
        print(f"verified cut points in {round(time.process_time() - start_time, 2)} s")

        # for cut_point in cut_points:
        #     print(cut_point)


if __name__ == "__main__":
    main()
