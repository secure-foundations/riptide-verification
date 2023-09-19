from typing import Dict

import json
from argparse import ArgumentParser

import semantics.smt as smt

from semantics.dataflow.graph import DataflowGraph
from semantics.dataflow.semantics import SymbolicExecutor, WORD_WIDTH
from semantics.dataflow.permission import MemoryPermissionSolver


def main():
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, help="Stop when <n> terminating configurations are found")
    parser.add_argument("dfg", help="A json file (.o2p) describing the dataflow graph")
    parser.add_argument("arg_assignment", nargs="*", help="Set function arguments to constant integers, e.g. len=10")
    args = parser.parse_args()

    with open(args.dfg) as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source))

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

    executor = SymbolicExecutor(dfg, free_vars)

    heap_objects = MemoryPermissionSolver.get_static_heap_objects(dfg)

    configs = [(executor.configurations[0], 0)]
    num_terminating_configs = 0

    # Doing a BFS on the state space
    while len(configs):
        config, num_steps = configs.pop(0)
        next_configs = executor.step(config)

        if len(next_configs) == 0:
            num_terminating_configs += 1

            print(f"terminating configuration #{num_terminating_configs} after {num_steps} step(s)")
            print(f"  path constraints: {config.path_constraints}")
            print("  memory updates:")
            for update in config.memory:
                print(f"    {update.base}[{update.index}] = {update.value}")
            
            # for i, channel in enumerate(config.channel_states):
            #     print(f"  channel {i}: {len(channel.values)} value(s)")

            print(f"  {len(config.permission_constraints)} permission constraint(s)")
            solution = MemoryPermissionSolver.solve_constraints(heap_objects, config.permission_constraints)
            # for constraint in config.permission_constraints:
            #     print(f"  {constraint}")
            if solution is None:
                print("unable to find consistent permission assignment, potential data race")
                break

            print("  found a permission solution")

            if args.n is not None and num_terminating_configs >= args.n:
                break

            # for var, term in solution.items():
            #     print(f"{var} = {term}")

        else:
            # print("#########################")
            # for i, channel in enumerate(next_configs[0].channel_states):
            #     print(f"  channel {i}: {len(channel.values)} value(s)")

            configs.extend(tuple(zip(next_configs, [num_steps + 1] * len(next_configs))))


if __name__ == "__main__":
    main()
