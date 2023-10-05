from typing import Dict

import json

import semantics.smt as smt

from semantics.matching import *
from semantics.dataflow.graph import DataflowGraph
from semantics.dataflow.semantics import NextConfiguration, StepException, Configuration, WORD_WIDTH, CarryOperator, PermissionedValue
from semantics.dataflow.permission import MemoryPermissionSolver, PermissionVariable


def main():
    with open("examples/bisim/test-1.o2p") as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source))

    free_vars: Dict[str, smt.SMTTerm] = {}

    for function_arg in dfg.function_arguments:
        free_vars[function_arg.variable_name] = smt.FreshSymbol(smt.BVType(WORD_WIDTH))

    print("function argument to SMT variables:", free_vars)

    config = Configuration.get_initial_configuration(dfg, free_vars)

    dummy_permission = PermissionVariable("dummy")

    # See examples/bisim/test-1.png for operator and channel IDs
    loop_header_2 = config.copy()
    loop_header_2.operator_states[0].transition_to(CarryOperator.pass_b)
    loop_header_2.channel_states[1].pop()

    # loop_header_2.channel_states[0].push(PermissionedValue(smt.FreshSymbol(smt.BVType(WORD_WIDTH), "icmp_%d"), dummy_permission))
    loop_header_2.channel_states[2].push(PermissionedValue(smt.FreshSymbol(smt.BVType(WORD_WIDTH), "inc_%d"), dummy_permission))

    configs = [(loop_header_2.copy(), 0)]

    schedule = [0, 4, 0, 1, 3, 2]

    while len(configs) != 0:
        config, num_steps = configs.pop(0)

        assert num_steps < len(schedule)
        results = config.step_exhaust(schedule[num_steps])

        for result in results:
            if isinstance(result, NextConfiguration):
                match = loop_header_2.match(result.config)
                if isinstance(match, MatchingSuccess):
                    print(f"### matching success at step {num_steps + 1}")
                    print(result.config)
                    print("substitution:", match.substitution)
                    print("condition:", match.condition.simplify())
                    continue

                configs.append((result.config, num_steps + 1))
            elif isinstance(result, StepException):
                assert False, f"got exception: {result.reason}"

        if len(results) == 0:
            print(f"terminating state at step {num_steps}")
            print(config)

    # while len(configs) != 0:
        # print(f"=============== {len(configs)}")
        # for config, _ in configs:
        #     print(config.path_constraints)

    #     config, num_steps = configs.pop(0)
    #     # print(config)

    #     # Try step each PE until hit a branch or exhausted
    #     changed = False
    #     for pe_info in dfg.vertices:
    #         results = config.step_exhaust(pe_info.id)

    #         if len(results) == 0:
    #             continue
    #         elif len(results) == 1:
    #             if isinstance(results[0], NextConfiguration):
    #                 changed = True
    #                 config = results[0].config
    #             elif isinstance(results[0], StepException):
    #                 assert False, f"got exception: {results[0].reason}"
    #         else:
    #             changed = True
    #             # print(f"branching on {pe_info.id}!", len(results))
    #             configs.extend((result.config, num_steps + 1) for result in results if isinstance(result, NextConfiguration))
    #             break
    #     else:
    #         # print("no branching")
    #         if changed:
    #             configs.append((config, num_steps + 1))

    #     if not changed:
    #         num_terminating_configs += 1

    #         print(f"terminating configuration #{num_terminating_configs} after {num_steps} step(s)")
    #         print(f"  path constraints: {config.path_conditions}")
    #         print("  memory updates:")
    #         for update in config.memory_updates:
    #             print(f"    {update.base}[{update.index}] = {update.value}")

    #         # Check memory permission constraints
    #         print(f"  {len(config.permission_constraints)} permission constraint(s)")
    #         solution = MemoryPermissionSolver.solve_constraints(heap_objects, config.permission_constraints)
    #         # for constraint in config.permission_constraints:
    #         #     print(f"  {constraint}")
    #         if solution is None:
    #             print("unable to find consistent permission assignment, potential data race")
    #             break

    #         print("  found a permission solution")


if __name__ == "__main__":
    main()
