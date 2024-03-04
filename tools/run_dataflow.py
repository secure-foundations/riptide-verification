from typing import Dict

import json
from argparse import ArgumentParser

import semantics.smt as smt

from semantics.dataflow import (
    DataflowGraph, NextConfiguration, StepException,
    Configuration, WORD_WIDTH,
)

from semantics.dataflow.permission import PermissionSolver, ResultUnsat, FiniteFractionalPA


def main():
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, help="Stop when <n> terminating configurations are found")
    parser.add_argument("--channel-bound", type=int, help="Buffer size at each channel (infinite if not specified)")
    parser.add_argument("--disable-confluence-check", action="store_const", const=True, default=False, help="Disable confluence check")
    parser.add_argument("dfg", help="A json file (.o2p) describing the dataflow graph")
    parser.add_argument("arg_assignment", nargs="*", help="Set function arguments to constant integers, e.g. len=10")
    args = parser.parse_args()

    with open(args.dfg) as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source), args.channel_bound)

    function_arg_names = set(arg.variable_name for arg in dfg.function_arguments)
    free_vars: Dict[str, smt.SMTTerm] = {}

    for assignment in args.arg_assignment:
        assert assignment.count("=") == 1, f"ill-formed assignment {assignment}"
        name, value = assignment.split("=")
        assert name in function_arg_names, f"argument {name} does not exist"
        free_vars[name] = smt.BV(int(value), WORD_WIDTH)

    for function_arg in dfg.function_arguments:
        if function_arg.variable_name not in free_vars:
            free_vars[function_arg.variable_name] = smt.FreshSymbol(smt.BVType(WORD_WIDTH))

    print("function argument to SMT variables:", free_vars)

    heap_objects = PermissionSolver.get_static_heap_objects(dfg)

    configs = [(Configuration.get_initial_configuration(dfg, free_vars, disable_permissions=args.disable_confluence_check), 0)]
    num_terminating_configs = 0

    # Doing a BFS on the state space
    while len(configs) != 0:
        # print(f"=============== {len(configs)}")
        # for config, _ in configs:
        #     print(config.path_constraints)

        config, num_steps = configs.pop(0)
        # print(config)

        # Try step each PE until hit a branch or exhausted
        changed = False
        for pe_info in dfg.vertices:
            results = config.step_exhaust(pe_info.id)

            if len(results) == 0:
                continue
            elif len(results) == 1:
                if isinstance(results[0], NextConfiguration):
                    changed = True
                    config = results[0].config
                elif isinstance(results[0], StepException):
                    assert False, f"got exception: {results[0].reason}"
            else:
                changed = True
                # print(f"branching on {pe_info.id}!", len(results))
                configs.extend((result.config, num_steps + 1) for result in results if isinstance(result, NextConfiguration))
                break
        else:
            # print("no branching")
            if changed:
                configs.append((config, num_steps + 1))

        if not changed:
            num_terminating_configs += 1

            print(f"terminating configuration #{num_terminating_configs} after {num_steps} step(s)")
            print(f"  path constraints: {config.path_conditions}")
            print("  memory updates:")
            for update in config.memory_updates:
                print(f"    {update.base}[{update.index}] = {update.value}")

            # Check memory permission constraints
            print(f"  {len(config.permission_constraints)} permission constraint(s)")

            if not args.disable_confluence_check:
                perm_algebra = FiniteFractionalPA(heap_objects, 4)
                result = PermissionSolver.solve_constraints(perm_algebra, config.permission_constraints)
                # for constraint in config.permission_constraints:
                #     print(f"  {constraint}")
                if isinstance(result, ResultUnsat):
                    print("unable to find consistent permission assignment, potential data race")
                    break
                # else:
                #     assert isinstance(result, ResultSat)
                #     for var, term in result.solution.items():
                #         print(f"{var} = {term}")

                print("  found a permission solution")


if __name__ == "__main__":
    main()
