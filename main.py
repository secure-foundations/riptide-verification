from typing import Dict

import json
from argparse import ArgumentParser

import smt

from dataflow import DataflowGraph
from symbolic import SymbolicExecutor
from permission import MemoryPermissionSolver


def main():
    parser = ArgumentParser()
    parser.add_argument("dfg", help="A json file (.o2p) describing the dataflow graph")
    parser.add_argument("-n", type=int, help="stop when <n> terminating configurations are found")
    args = parser.parse_args()

    with open(args.dfg) as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source))

    free_vars: Dict[str, smt.SMTTerm] = {
        function_arg.variable_name: smt.FreshSymbol(smt.INT)
        for function_arg in dfg.function_arguments
    }
    print("function argument to SMT variables:", free_vars)

    executor = SymbolicExecutor(dfg, free_vars)

    heap_objects = MemoryPermissionSolver.get_static_heap_objects(dfg)

    configs = [executor.configurations[0]]
    num_terminating_configs = 0

    while len(configs):
        config = configs.pop(0)
        next_configs = executor.step(config)

        if len(next_configs) == 0:
            num_terminating_configs += 1

            print(f"terminating configuration #{num_terminating_configs}")
            print(f"  path constraints: {config.path_constraints}")
            print("  memory updates:")
            for update in config.memory:
                print(f"    {update.base}[{update.index}] = {update.value}")
            
            # for i, channel in enumerate(config.channel_states):
            #     print(f"  channel {i}: {len(channel.values)} value(s)")

            solution = MemoryPermissionSolver.solve_constraints(heap_objects, config.permission_constraints)
            # for constraint in config.permission_constraints:
            #     print(f"  {constraint}")
            if solution is None:
                print("unable to find consistent permission assignment, potential data race")
                break

            if args.n is not None and num_terminating_configs >= args.n:
                break

            # for var, term in solution.items():
            #     print(f"{var} = {term}")

        else:
            configs.extend(next_configs)


if __name__ == "__main__":
    main()
