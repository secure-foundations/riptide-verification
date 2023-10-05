from typing import Dict

import json

import semantics.smt as smt
from semantics.matching import *

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def main():
    """
    See examples/bisim/test-1.png for operator and channel IDs in the dataflow graph
    """

    with open("examples/test-1/test-1.o2p") as dataflow_source:
        dfg = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
        # Set up initial config for the dataflow program
        free_vars: Dict[str, smt.SMTTerm] = {
            function_arg.variable_name: smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_{function_arg.variable_name}_%d")
            for function_arg in dfg.function_arguments
        }
        dataflow_init_config = dataflow.Configuration.get_initial_configuration(dfg, free_vars)

        # Set up the invariant state
        # TODO: ignoring memory permissions for now
        dummy_permission = dataflow.PermissionVariable("dummy")
        dataflow_invariant_config = dataflow_init_config.copy()
        dataflow_invariant_config.operator_states[0].transition_to(dataflow.CarryOperator.pass_b)
        dataflow_invariant_config.channel_states[1].pop()
        dataflow_invariant_config.channel_states[2].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_inc_%d"),
                dummy_permission,
            ),
        )

    with open("examples/test-1/test-1.ll") as f:
        module = llvm.Parser.parse_module(f.read())
        function = tuple(module.functions.values())[0]
    
        llvm_init_config = llvm.Configuration.get_initial_configuration(module, function)
        llvm_invariant_config = llvm.Configuration(
            module,
            function,
            current_block="header",
            previous_block="body",
            current_instr_counter=0,
            variables=OrderedDict([
                (r"%A", smt.FreshSymbol(smt.BVType(64), "llvm_A_%d")),
                (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_len_%d")),
                (r"%inc", smt.FreshSymbol(smt.BVType(32), "llvm_inc_%d")),
            ]),
            path_conditions=[],
        )

    # Want to check:
    # from dataflow_init_config or dataflow_invariant_config, we can reach either the end or dataflow_invariant_config
    # from llvm_init_config or llvm_invariant_config, we can reach either the end or llvm_invariant_config
    # and further more these states "match with each other"

    dataflow_cut_points = [dataflow_init_config, dataflow_invariant_config]
    llvm_cut_points = [llvm_init_config, llvm_invariant_config]

    # TODO: automatically infer this from the LLVM execution trace
    some_schedule = [ 0, 4, 0, 1, 3, 2 ]

    for i, (dataflow_cut_point, llvm_cut_point) in enumerate(zip(dataflow_cut_points, llvm_cut_points)):
        print(f"##### trying cut point pair {i} #####")

        # configs matched to the invariant config
        matched_dataflow_configs = []
        matched_llvm_configs = []

        final_dataflow_configs = []
        final_llvm_configs = []  
        
        # First try executing the dataflow cut point
        queue = [(dataflow_cut_point.copy(), 0)]
        while len(queue) != 0:
            config, num_steps = queue.pop(0)

            assert num_steps < len(some_schedule), "dataflow execution goes beyond the existing schedule"
            results = config.step_exhaust(some_schedule[num_steps])

            for result in results:
                if isinstance(result, dataflow.NextConfiguration):
                    match = dataflow_invariant_config.match(result.config)
                    if isinstance(match, MatchingSuccess):
                        print(f"[dataflow] found a matched config at step {num_steps + 1}")
                        # TODO: check match condition here
                        matched_dataflow_configs.append((result.config, match))
                    else:
                        queue.append((result.config, num_steps + 1))

                elif isinstance(result, dataflow.StepException):
                    assert False, f"got exception: {result.reason}"

            if len(results) == 0:
                print(f"[dataflow] found a final config at step {num_steps}")
                final_dataflow_configs.append(config)

        # Then try the llvm cut point
        queue = [llvm_cut_point.copy()]
        while len(queue) != 0:
            config = queue.pop(0)
            results = config.step()

            for result in results:
                if isinstance(result, llvm.NextConfiguration):
                    match = llvm_invariant_config.match(result.config)
                    if isinstance(match, MatchingSuccess):
                        print(f"[llvm] found a matched config")
                        matched_llvm_configs.append((result.config, match))
                    else:
                        queue.append(result.config)

                elif isinstance(result, llvm.FunctionReturn):
                    print(f"[llvm] found a final config")
                    final_llvm_configs.append(config.copy())

                else:
                    assert False, f"unsupported result {result}"
    
        print("### matched configs")
        for config, match in matched_dataflow_configs:
            print(config)
            print("matching substitution:", match.substitution)
            print("matching condition:", match.condition.simplify())
        
        for config, match in matched_llvm_configs:
            print(config)
            print("matching substitution:", match.substitution)
            print("matching condition:", match.condition.simplify())

        print("### final configs")
        for config in final_dataflow_configs:
            print(config)
        
        for config in final_llvm_configs:
            print(config)

if __name__ == "__main__":
    main()
