from typing import Callable

import json
import argparse

from semantics.dataflow import *


"""
Schedule is a function that picks a list of fireable
operators in any configuration to explore (in parallel)
"""
OperatorSchedule = Callable[[Configuration], Tuple[int, ...]]


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
) -> Tuple[Configuration]:
    """
    Explore the tree of symbolic states from an initial configuration,
    up to a certain depth
    """

    # [(config, depth)]
    queue: List[Tuple[Configuration, int]] = [(initial, 0)]

    explored_states: List[Configuration] = []

    max_branching: int = 1

    # BFS on the symbolic configurations
    while len(queue) != 0:
        config, depth = queue.pop(0)
        explored_states.append(config)

        if depth >= max_depth:
            continue

        operators_to_fire = schedule(config)

        num_branches = 0

        for pe_id in operators_to_fire:
            results = config.copy().step(pe_id)

            for result in results:
                assert isinstance(result, NextConfiguration)
                queue.append((result.config, depth + 1))
                num_branches += 1

        if num_branches > max_branching:
            max_branching = num_branches

    print("max branching", max_branching)

    return tuple(explored_states)


def partition_relation(config1: Configuration, config2: Configuration) -> bool:
    assert config1.graph == config2.graph

    for operator in config1.graph.vertices:
        op_state1 = config1.operator_states[operator.id]
        op_state2 = config2.operator_states[operator.id]

        if op_state1.current_transition != op_state2.current_transition:
            return False

    for channel in config1.graph.channels:
        channel_state1 = config1.channel_states[channel.id]
        channel_state2 = config2.channel_states[channel.id]

        if (channel_state1.hold_constant is not None and
            channel_state2.hold_constant is not None):
            continue

        if channel_state1.count() != channel_state2.count():
            return False

    return True


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
            # TODO: use this to infer a more exact invariant
            # values_in_partition = [ config.operator_states[pe.id].value for config in partition ]

            if cut_point.operator_states[pe.id].value is not None:
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
            if (len(set(values_in_partition)) == 1 and
                (unique_value.is_symbol() or unique_value.is_constant())):

                value = PermissionedValue(unique_value, dummy_perm)

                # print("unique", values_in_partition[0])
            else:
                # print("values", set(values_in_partition))

                value = PermissionedValue(
                    smt.FreshSymbol(smt.BVType(WORD_WIDTH), f"cut_point_channel_{channel.id}_{index}_%d"),
                    dummy_perm,
                )

            cut_point.channel_states[channel.id].values[index] = value

    return cut_point


def construct_cut_point_abstraction(
    # cut_points: Tuple[Configuration, ...],
    configs: Tuple[Configuration, ...],
    schedule: OperatorSchedule,
    # offset: int = 0,
) -> Tuple[Configuration, ...]:
    """
    Check if the given set of cut points is a valid abstraction of the concrete semantics
    (that is they are closed under execution)

    If so, return the same set
    Otherwise, we may append new cut points and generate a final version
    """

    partitions: List[List[Configuration]] = []

    # Partition the configurations according to their "shape" (i.e. number of values in each channel)
    for config in configs:
        for partition in partitions:
            if partition_relation(config, partition[0]):
                partition.append(config)
                break
        else:
            partitions.append([config])

    # Construct a cut point for each partition
    cut_points = list(generalize_partition(partition) for partition in partitions)

    cut_points_to_check = list(range(len(cut_points)))

    while len(cut_points_to_check) != 0:
        unmatched_configs: List[Configuration] = []

        remaining_cut_points_to_check: Set[int] = set()

        for i in cut_points_to_check:
            # print(f"checking cut point {i}")

            cut_point = cut_points[i]
            operators_to_fire = schedule(cut_point)

            for pe_id in operators_to_fire:
                results = cut_point.copy().step(pe_id)

                # match against another cut points
                for result in results:
                    assert isinstance(result, NextConfiguration)

                    for j, other_cut_point in enumerate(cut_points):
                        # print(f"matching against cut point {j}")
                        # if partition_relation(result.config, other_cut_point):
                        #     match_result, _ = other_cut_point.match(result.config)

                        #     # print(result.config, other_cut_point)
                        #     # print(isinstance(match_result, MatchingSuccess), match_result.check_condition())

                        #     # if not isinstance(match_result, MatchingSuccess):
                        #     #     print(match_result)

                        #     assert isinstance(match_result, MatchingSuccess) and match_result.check_condition()

                        match_result, _ = other_cut_point.match(result.config)
                        if isinstance(match_result, MatchingSuccess) and match_result.check_condition():
                            # print(match_result.condition.to_smtlib())
                            # assert match_result.check_condition()
                            break

                        # if isinstance(match_result, MatchingSuccess) and not match_result.check_condition():
                        #     print("condition failed", match_result.condition.to_smtlib())

                    else:
                        # print(cut_point)
                        # print(result.config)
                        unmatched_configs.append(result.config)
                        # remaining_cut_points_to_check.add(i)

        # Still found configurations that are not matched
        # Refine partition and cut points again
        if len(unmatched_configs) != 0:
            # which partitions got updated
            updated_partition_indices: Set[int] = set()

            num_new = 0

            for config in unmatched_configs:
                for i, partition in enumerate(partitions):
                    if partition_relation(config, partition[0]):
                        partition.append(config)
                        remaining_cut_points_to_check.add(i)
                        updated_partition_indices.add(i)
                        break
                else:
                    updated_partition_indices.add(len(partitions))
                    remaining_cut_points_to_check.add(len(partitions))
                    partitions.append([config])
                    cut_points.append(None) # placeholder
                    num_new += 1

            for i in updated_partition_indices:
                cut_points[i] = generalize_partition(partitions[i])

            print(
                f"found {len(unmatched_configs)} unmatched config(s), "
                f"{num_new} new partition(s), "
                f"{len(updated_partition_indices) - num_new} updated partition(s), "
                f"total {len(partitions)} partitions"
            )

            cut_points_to_check = remaining_cut_points_to_check
        else:
            break

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
    #             results = cut_point.copy().step(pe_id)

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

    initial = Configuration.get_initial_configuration(dfg, free_vars, disable_permissions=True)

    configs = [initial] # explore_states(initial, pure_priority_schedule, 1)

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

    print(f"graph size: {len(dfg.vertices)} operator(s), {len(dfg.channels)} channel(s)")

    cut_points = construct_cut_point_abstraction(configs, pure_priority_schedule)

    print("total number of cut points: ", len(cut_points))


if __name__ == "__main__":
    main()
