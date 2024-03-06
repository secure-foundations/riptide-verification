from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from collections import OrderedDict

import json
import time
import argparse

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

    def get_incoming_shapes(self, shape: ConfigurationShape) -> Tuple[ConfigurationShape]:
        shape_index = self.shape_to_index[shape]
        indices = sorted(self.incoming_edges.get(shape_index, set()))
        return tuple(self.shapes[index] for index in indices)

    def get_outgoing_shapes(self, shape: ConfigurationShape) -> Tuple[ConfigurationShape]:
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
    """

    graph: DataflowGraph
    dependency: Dict[int, Set[int]] = field(default_factory=dict)

    def add_dependencies(self, pe_id1: int, pe_id2: int):
        if pe_id1 not in self.dependency:
            self.dependency[pe_id1] = set()
        self.dependency[pe_id1].add(pe_id2)

    @staticmethod
    def from_shape(shape: ConfigurationShape) -> FirebilityDependencyGraph:
        ...

    def find_loop(self) -> Tuple[int, ...]:
        """
        Find one loop in the graph
        """


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
    start_time = time.time()

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
                    f"total time {round(time.time() - start_time, 2)} s"
                )
                print(*zip(other_fan_ins, other_fan_outs))

            if depth >= max_depth:
                continue

            operators_to_fire = schedule(config)

            num_branches = 0

            for pe_id in operators_to_fire:
                start = time.time()
                results = config.copy().step_exhaust(pe_id)
                symbolic_execution_time += time.time() - start

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

    for pe in cut_point.graph.vertices:
        # The transition is already set, we now need to infer internal state values
        # if they are required (e.g. for Invariant)

        if pe.operator == "CF_CFG_OP_INVARIANT":
            assert isinstance(cut_point.operator_states[pe.id], InvariantOperator)

            if cut_point.operator_states[pe.id].value is not None:
                # TODO: use this to infer a more exact invariant
                values_in_partition = [ config.operator_states[pe.id].value for config in partition ]

                # unique_value = values_in_partition[0]
                # if (len(set(values_in_partition)) == 1 and
                #     unique_value.get_free_variables().issubset(function_arg_free_vars)):
                #     value = unique_value
                # else:
                #     value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_inv_{pe.id}_%d")

                # cut_point.operator_states[pe.id].value = value
                cut_point.operator_states[pe.id].value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_inv_{pe.id}_%d")

    for channel in cut_point.graph.channels:
        if channel.hold or channel.constant is not None:
            continue

        num_values = cut_point.channel_states[channel.id].count()

        # Clear all values for now
        # cut_point.channel_states[channel.id].values = []

        for index in range(num_values):
            # TODO: use this to infer a more exact invariant
            values_in_partition = [ config.channel_states[channel.id].values[index].term for config in partition ]

            unique_value = values_in_partition[0]

            # could_be_non_zero = False
            # could_be_non_one = False

            # for config in partition:
            #     value = config.channel_states[channel.id].values[index].term

            #     with smt.push_solver(config.solver):
            #         for condition in config.path_conditions:
            #             config.solver.add_assertion(condition)

            #         with smt.push_solver(config.solver):
            #             config.solver.add_assertion(smt.Not(smt.Equals(value, smt.BVConst(0, WORD_WIDTH))))

            #             if config.solver.solve():
            #                 could_be_non_zero = True

            #         with smt.push_solver(config.solver):
            #             config.solver.add_assertion(smt.Not(smt.Equals(value, smt.BVConst(1, WORD_WIDTH))))

            #             if config.solver.solve():
            #                 could_be_non_one = True

            #         if could_be_non_zero and could_be_non_one:
            #             break

            # assert could_be_non_zero or could_be_non_one

            # if not could_be_non_zero:
            #     value = smt.BVConst(0, WORD_WIDTH)

            # elif not could_be_non_one:
            #     value = smt.BVConst(1, WORD_WIDTH)

            # else:
            #     value = smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_channel_{channel.id}_{index}_%d")

            # if len(set(values_in_partition)):
            #     print(unique_value.get_free_variables(), unique_value.get_free_variables().issubset(function_arg_free_vars))

            # if (len(set(values_in_partition)) == 1 and
            #     unique_value.get_free_variables().issubset(function_arg_free_vars)):
            #     value = PermissionedValue(unique_value, dummy_perm)
            # else:
            #     value = PermissionedValue(
            #         smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_channel_{channel.id}_{index}_%d"),
            #         dummy_perm,
            #     )

            # cut_point.channel_states[channel.id].values[index] = value

            cut_point.channel_states[channel.id].values[index] = PermissionedValue(
                smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_channel_{channel.id}_{index}_%d"),
                dummy_perm,
            )

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

    # Infer some constraints about whether a value has to be zero or non-zero
    for channel in cut_point.graph.channels:
        if channel.hold:
            continue

        for index in range(cut_point.channel_states[channel.id].count()):

            cut_point_value = cut_point.channel_states[channel.id].values[index].term

            all_zeroes = True
            all_non_zeroes = True

            for config in partition:
                value = config.channel_states[channel.id].values[index].term

                with smt.push_solver(config.solver):
                    for condition in config.path_conditions:
                        config.solver.add_assertion(condition)

                    if all_zeroes:
                        with smt.push_solver(config.solver):
                            config.solver.add_assertion(smt.Not(smt.Equals(value, smt.BVConst(0, WORD_WIDTH))))

                            if config.solver.solve():
                                all_zeroes = False
                            else:
                                all_non_zeroes = False

                    if all_non_zeroes:
                        with smt.push_solver(config.solver):
                            config.solver.add_assertion(smt.Equals(value, smt.BVConst(0, WORD_WIDTH)))

                            if config.solver.solve():
                                all_non_zeroes = False
                            else:
                                all_zeroes = False

                if not all_zeroes and not all_non_zeroes:
                    break

            assert not (all_zeroes and all_non_zeroes)

            if all_zeroes:
                cut_point.path_conditions.append(smt.Equals(cut_point_value, smt.BVConst(0, WORD_WIDTH)))

            elif all_non_zeroes:
                cut_point.path_conditions.append(smt.Not(smt.Equals(cut_point_value, smt.BVConst(0, WORD_WIDTH))))

    # for config in partition:
    #     match_result, _ = cut_point.match(config)
    #     assert isinstance(match_result, MatchingSuccess)

    #     if not match_result.check_condition():
    #         print(cut_point, config)
    #         print(match_result.condition)
    #         exit()

    return cut_point


def construct_cut_point_abstraction(
    # cut_points: Tuple[Configuration, ...],
    configs: Tuple[Configuration, ...],
    schedule: OperatorSchedule,
    exact_partitions: ShapePartition,
    # offset: int = 0,
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

            cut_point = cut_points[shape]
            operators_to_fire = schedule(cut_point)

            for pe_id in operators_to_fire:
                results = cut_point.copy().step_exhaust(pe_id)

                # match against another cut points
                for result in results:
                    assert isinstance(result, NextConfiguration)

                    # if (ConfigurationShape.from_config(result.config) not in exact_partitions and
                    #     ConfigurationShape.from_config(cut_point) in exact_partitions):
                    #     print(cut_point)
                    #     print(result.config)
                    #     exit()

                    for other_cut_point in cut_points.values():
                        match_result, _ = other_cut_point.match(result.config)
                        if isinstance(match_result, MatchingSuccess) and match_result.check_condition():
                            # print(match_result.condition.to_smtlib())
                            # assert match_result.check_condition()
                            break

                        # if isinstance(match_result, MatchingSuccess) and not match_result.check_condition():
                        #     print("condition failed", match_result.condition.to_smtlib())

                    else:
                        unmatched_configs.append(result.config)

        shapes_to_check = []

        # Still found configurations that are not matched
        # Refine partition and cut points again
        if len(unmatched_configs) != 0:
            num_new = 0

            updated_shapes: OrderedDict[ConfigurationShape, None] = OrderedDict()

            for config in unmatched_configs:
                shape = ConfigurationShape.from_config(config)
                if shape not in partitions:
                    num_new += 1

                updated_shapes[shape] = None
                partitions.add(config)

            shapes_to_check = list(updated_shapes.keys())

            for shape in shapes_to_check:
                cut_points[shape] = generalize_partition(partitions[shape])

            print(
                f"found {len(unmatched_configs)} unmatched config(s), "
                f"{num_new} new partition(s), "
                f"{len(shapes_to_check) - num_new} updated partition(s), "
                f"total {len(partitions)} partition(s)"
            )

    return tuple(cut_points)

    # print(f"trying to construct cut points with {len(configs)} configurations and {len(partitions)} partition(s)")

    # # Check if the current set of cut points is sufficient
    # # If not, we would find a set of unmatched configurations
    # unmatched_configs = []
    # for i, cut_point in enumerate(cut_points):
    #     # if i < offset:
    #     #     continue

    #     # print(f"checking closedness of cut point {i}")

    #     queue: List[Configuration] = [cut_point]

    #     # BFS from the cut point
    #     while len(queue) != 0:
    #         config = queue.pop(0)

    #         operators_to_fire = schedule(config)

    #         for pe_id in operators_to_fire:
    #             results = cut_point.copy().step_exhaust(pe_id)

    #             # match against another cut points
    #             for result in results:
    #                 assert isinstance(result, NextConfiguration)

    #                 for j, other_cut_point in enumerate(cut_points):
    #                     # print(f"matching against cut point {j}")
    #                     match_result, _ = other_cut_point.match(result.config)
    #                     if isinstance(match_result, MatchingSuccess) and match_result.check_condition():
    #                         # print(match_result.condition.to_smtlib())
    #                         # assert match_result.check_condition()
    #                         break

    #                     # if isinstance(match_result, MatchingSuccess) and not match_result.check_condition():
    #                     #     print("condition failed", match_result.condition.to_smtlib())

    #                     # print(f"not matched to cut point {j}:", match_result.reason)
    #                 else:
    #                     # Continue execution
    #                     # print(result.config)

    #                     # print(cut_point)
    #                     # print(result.config)

    #                     unmatched_configs.append(result.config)

    #                     # assert False, "not matched after one step"
    #                     # print("not matched after one step, adding a new cut point")

    #                     # queue.append(result.config)

    # if len(unmatched_configs) != 0:
    #     print(f"found {len(unmatched_configs)} unmatched configs")

    #     # print(unmatched_configs[0])

    #     # # Put the config
    #     # for config in unmatched_configs:
    #     #     for partition in partitions:
    #     #         if partition_relation(config, partition[0]):
    #     #             partition.append(config)
    #     #             break
    #     #     else:
    #     #         partitions.append([config])

    #     # new_cut_points = tuple(generalize_partition(partition) for partition in partitions)

    #     # Retry with more configs
    #     return construct_cut_point_abstraction(configs + tuple(unmatched_configs), schedule)
    # else:
    #     return cut_points


def check_deadlock(shape: ConfigurationShape) -> Tuple[int, ...]:
    """
    Return which operators are in a deadlock
    """




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

    with smt.Solver("z3") as solver:
        initial = Configuration.get_initial_configuration(dfg, free_vars, disable_permissions=True, solver=solver)

        # partitions: List[List[Configuration]] = []

        # # TODO: inefficient partition algo
        # for state in configs:
        #     for partition in partitions:
        #         if partition_relation(state, partition[0]):
        #             partition.append(state)
        #             break
        #     else:
        #         partitions.append([state])

        # for state in states:
        #     print(state)

        # sorted_partitions = sorted(partitions, key=lambda p: len(p), reverse=True)
        # for config in sorted_partitions[0]:
        #     print(config)

        # print(len(configs), len(partitions))

        # cut_points = tuple(generalize_partition(partition) for partition in partitions)

        # # cut_points.pop(3)
        # # cut_points.pop(2)
        # # cut_points.pop(1)

        # cut_points = construct_cut_point_abstraction(cut_points, pure_priority_schedule)

        # print(f"graph size: {len(dfg.vertices)} operator(s), {len(dfg.channels)} channel(s)")

        # _, partitions = explore_states(initial, pure_priority_schedule, 1000)

        # for part in partitions.shape_map.values():
        #     print(part[0])

        cut_points = construct_cut_point_abstraction([initial], pure_priority_schedule, None)

        print("total number of cut points: ", len(cut_points))


if __name__ == "__main__":
    main()


"""

C

C/~, c ~ c' iff they have the same shape

For each partition { c_1, ..., c_n } in C/~, infer a cut point p such that |p| >= |c_i| for each i

{ p_1, ..., p_m }

"""
