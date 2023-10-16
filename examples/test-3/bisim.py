from typing import Dict

import json

import semantics.smt as smt
from semantics.matching import *

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def main():
    """
    See examples/bisim/test-3.png for operator and channel IDs in the dataflow graph
    """

    with open("examples/test-3/test-3.o2p") as dataflow_source:
        dfg = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
        # Set up initial config for the dataflow program
        free_vars: Dict[str, smt.SMTTerm] = {
            function_arg.variable_name: smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_param_{function_arg.variable_name}_%d")
            for function_arg in dfg.function_arguments
        }
        dataflow_init_config = dataflow.Configuration.get_initial_configuration(dfg, free_vars)

        dummy_permission = dataflow.PermissionVariable("dummy")

        # dataflow_outer_1_config = dataflow_init_config.copy()
        dataflow_outer_2_config = dataflow_init_config.copy()
        # dataflow_inner_1_config = dataflow_init_config.copy()
        dataflow_inner_2_config = dataflow_init_config.copy()

        # outer loop invariant state
        dataflow_outer_2_config.operator_states[11].transition_to(dataflow.CarryOperator.loop)
        dataflow_outer_2_config.operator_states[2].transition_to(dataflow.CarryOperator.loop)

        dataflow_outer_2_config.channel_states[26].pop()
        dataflow_outer_2_config.channel_states[5].pop()

        dataflow_outer_2_config.channel_states[25].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_outer_2_config.channel_states[4].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_outer_2_config.channel_states[27].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_inc_i_%d"),
                dummy_permission,
            ),
        )
        dataflow_outer_2_config.channel_states[6].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_lso_alloc1_0_%d"),
                dummy_permission,
            ),
        )

        # inner loop invariant state
        dataflow_inner_2_config.operator_states[11].transition_to(dataflow.CarryOperator.loop)
        dataflow_inner_2_config.operator_states[2].transition_to(dataflow.CarryOperator.loop)
        dataflow_inner_2_config.operator_states[5].transition_to(dataflow.CarryOperator.loop)
        dataflow_inner_2_config.operator_states[17].transition_to(dataflow.CarryOperator.loop)
        
        dataflow_inner_2_config.operator_states[4].transition_to(dataflow.InvariantOperator.loop)
        dataflow_inner_2_config.operator_states[4].value = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_i_0_%d")

        dataflow_inner_2_config.channel_states[26].pop()
        dataflow_inner_2_config.channel_states[5].pop()

        dataflow_inner_2_config.channel_states[25].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_inner_2_config.channel_states[4].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_inner_2_config.channel_states[39].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_inner_2_config.channel_states[11].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_inner_2_config.channel_states[9].push(
            dataflow.PermissionedValue(
                smt.BVConst(1, dataflow.WORD_WIDTH),
                dummy_permission,
            ),
        )
        dataflow_inner_2_config.channel_states[13].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_1_%d"),
                dummy_permission,
            ),
        )
        # TODO: needs a path condition to say this is equal to dataflow_inner_2_config.operator_states[4].value
        dataflow_inner_2_config.channel_states[21].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_i_0_%d"),
                dummy_permission,
            ),
        )
        dataflow_inner_2_config.channel_states[41].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_inc_j_%d"),
                dummy_permission,
            ),
        )

    #     # Set up the invariant state
    #     # TODO: ignoring memory permissions for now
    #     dummy_permission = dataflow.PermissionVariable("dummy")
    #     dataflow_invariant_config = dataflow_init_config.copy()
    #     dataflow_invariant_config.operator_states[1].transition_to(dataflow.CarryOperator.loop)
    #     dataflow_invariant_config.operator_states[2].transition_to(dataflow.InvariantOperator.loop)
    #     dataflow_invariant_config.operator_states[2].value = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_b_%d")

    #     dataflow_invariant_config.channel_states[0].pop()
    #     dataflow_invariant_config.channel_states[1].pop()
    #     dataflow_invariant_config.channel_states[2].push(
    #         dataflow.PermissionedValue(
    #             smt.BVConst(1, dataflow.WORD_WIDTH),
    #             dummy_permission,
    #         ),
    #     )
    #     dataflow_invariant_config.channel_states[3].pop()
    #     dataflow_invariant_config.channel_states[4].push(
    #         dataflow.PermissionedValue(
    #             smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_inc_%d"),
    #             dummy_permission,
    #         ),
    #     )
    #     dataflow_invariant_config.channel_states[5].push(
    #         dataflow.PermissionedValue(
    #             smt.BVConst(1, dataflow.WORD_WIDTH),
    #             dummy_permission,
    #         ),
    #     )

    with open("examples/test-3/test-3.lso.ll") as llvm_source:
        module = llvm.Parser.parse_module(llvm_source.read())
        function = tuple(module.functions.values())[0]
    
        llvm_init_config = llvm.Configuration.get_initial_configuration(module, function)
        
        # llvm_outer_1_config = llvm.Configuration(
        #     module,
        #     function,
        #     current_block="outer.header",
        #     previous_block="entry",
        #     current_instr_counter=0,
        #     variables=OrderedDict([
        #         (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_A_%d")),
        #         (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_B_%d")),
        #         (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_len_%d")),
        #     ]),
        #     path_conditions=[],
        # )

        llvm_outer_2_config = llvm.Configuration(
            module,
            function,
            current_block="outer.header",
            previous_block="outer.cleanup",
            current_instr_counter=0,
            variables=OrderedDict([
                (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_A_%d")),
                (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_B_%d")),
                (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_param_len_%d")),
                (r"%lso.alloc1.1.lcssa", smt.FreshSymbol(smt.BVType(32), "llvm_var_lso_alloc1_1_lcssa_%d")),
                (r"%inc.i", smt.FreshSymbol(smt.BVType(32), "llvm_var_inc_i_%d")),
            ]),
            path_conditions=[],
        )

        # llvm_inner_1_config = llvm.Configuration(
        #     module,
        #     function,
        #     current_block="inner.header",
        #     previous_block="outer.body",
        #     current_instr_counter=0,
        #     variables=OrderedDict([
        #         (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_A_%d")),
        #         (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_B_%d")),
        #         (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_len_%d")),
        #         (r"%lso.alloc1.0", smt.FreshSymbol(smt.BVType(32), "llvm_lso_alloc1_0_%d")),
        #         (r"%i", smt.FreshSymbol(smt.BVType(32), "llvm_i_%d")),
        #     ]),
        #     path_conditions=[],
        # )

        llvm_inner_2_config = llvm.Configuration(
            module,
            function,
            current_block="inner.header",
            previous_block="inner.body",
            current_instr_counter=0,
            variables=OrderedDict([
                (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_A_%d")),
                (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_B_%d")),
                (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_param_len_%d")),
                (r"%1", smt.FreshSymbol(smt.BVType(32), "llvm_var_1_%d")),
                (r"%inc.j", smt.FreshSymbol(smt.BVType(32), "llvm_var_inc_j_%d")),
                (r"%i", smt.FreshSymbol(smt.BVType(32), "llvm_var_i_%d")),
            ]),
            path_conditions=[],
        )

    dataflow_cut_points = [
        (dataflow_init_config, [
            2, # outer.header:0
            11, # outer.header:1
            0, # outer.header:2
            1, 15, 3, # triggered CF operators
            4, 17, 5, # triggered CF operators (second layer)
            6, # inner.header:2
            13, 12, 8, 10, # triggered CF operators
            (14, 9), # try both inner.body:2 or outer.cleanup:1
            (7, 4), # inner.body:3
            (16, 5), # inner.body:4
            17, # inner.header:1
        ]),
        (dataflow_outer_2_config, [
            2, # outer.header:0
            11, # outer.header:1
            0, # outer.header:2
            1, 15, 3, # triggered CF operators
            4, 17, 5, # triggered CF operators (second layer)
            6, # inner.header:2
            13, 12, 8, 10, # triggered CF operators
            (14, 9), # try inner.body:2 or outer.cleanup:1
            (7, 4), # try inner.body:3 or Inv_T at 4
            (16, 5), # try inner.body:4 or inner.header:0
            17, # inner.header:1
        ]),
        (dataflow_inner_2_config, [
            # 4, 17, 5, # triggered CF operators (second layer)
            # 6, # inner.header:2
            # 13, 12, 8, 10, # triggered CF operators
            # 14, # inner.body:2
            # 7, # inner.body:3
            # 16, # inner.body:4
            4, 17, 5, # triggered CF operators (second layer)
            6, # inner.header:2
            13, 12, 8, 10, # triggered CF operators
            (14, 9), # try inner.body:2 or outer.cleanup:1
            (7, 4), # try inner.body:3 or Inv_T at 4
            (16, 5), # try inner.body:4 or inner.header:0
            17, # inner.header:1
        ]),
    ]

    llvm_cut_points = [
        llvm_init_config,
        # llvm_outer_1_config,
        llvm_outer_2_config,
        # llvm_inner_1_config,
        llvm_inner_2_config,
    ]

    for i, ((dataflow_cut_point, schedule), llvm_cut_point) in enumerate(zip(dataflow_cut_points, llvm_cut_points)):

        print(f"##### trying cut point pair {i} #####")

        # configs matched to the invariant config

        # cut point index |-> configs matched to the cut point
        matched_dataflow_configs = { i: [] for i in range(len(llvm_cut_points)) }
        matched_llvm_configs = { i: [] for i in range(len(dataflow_cut_points)) }

        final_dataflow_configs = []
        final_llvm_configs = []

        # First try executing the dataflow cut point
        queue = [(dataflow_cut_point.copy(), 0)]
        while len(queue) != 0:
            config, num_steps = queue.pop(0)

            if num_steps >= len(schedule):
                print(config)

            assert num_steps < len(schedule), "dataflow execution goes beyond the existing schedule"
            
            # if num_steps == 16:
            #     print(config)

            next_operator = schedule[num_steps]

            if isinstance(next_operator, tuple):
                for op in next_operator:                
                    results = config.copy().step_exhaust(op)
                    if len(results) != 0:
                        break
            else:
                results = config.step_exhaust(next_operator)
   
            # print(f"[dataflow] === trying config after step {num_steps} (|next configs| = {len(results)})")

            for result in results:
                if isinstance(result, dataflow.NextConfiguration):
                    # if num_steps == 10:
                    #     print(result.config)

                    # if num_steps == 15:
                    #     print(result.config)

                    for j, (dataflow_cut_point, _) in enumerate(dataflow_cut_points):
                        # print(f"[dataflow] trying to match with cut point {j}")
                        match = dataflow_cut_point.match(result.config)
                        if isinstance(match, MatchingSuccess):
                            print(f"[dataflow] !!! found a matched config at step {num_steps + 1} to cut point {j}")
                            # TODO: check match condition here
                            matched_dataflow_configs[j].append((result.config, match))
                            break
                        # print(f"[dataflow] matching failed: {match.reason}")
                    else:
                        queue.append((result.config, num_steps + 1))

                elif isinstance(result, dataflow.StepException):
                    assert False, f"got exception: {result.reason}"

            if len(results) == 0:
                print(f"[dataflow] !!! found a final config at step {num_steps}")
                final_dataflow_configs.append(config)

        # Then try the llvm cut point
        queue = [llvm_cut_point.copy()]
        while len(queue) != 0:
            config = queue.pop(0)
            results = config.step()

            for result in results:
                if isinstance(result, llvm.NextConfiguration):
                    for j, llvm_cut_point in enumerate(llvm_cut_points):
                        match = llvm_cut_point.match(result.config)
                        if isinstance(match, MatchingSuccess):
                            print(f"[llvm] found a matched config to cut point {j}")
                            matched_llvm_configs[j].append((result.config, match))
                            break
                    else:
                        queue.append(result.config)

                elif isinstance(result, llvm.FunctionReturn):
                    print(f"[llvm] found a final config")
                    final_llvm_configs.append(config.copy())

                else:
                    assert False, f"unsupported result {result}"

        for j in range(len(dataflow_cut_points)):
            print(f"### matched configs at cut point {j}")
            for config, match in matched_dataflow_configs[j]:
                # print(config)
                assert match.check_condition(), "invalid match"
                print("===== dataflow start =====")
                print("memory_updates:")
                for update in config.memory_updates:
                    print(f"  {update}")
                print("path conditions:")
                for path_condition in config.path_conditions:
                    print(f"  {path_condition}")
                print("matching substitution:", match.substitution)
                print("===== dataflow end =====")

            for config, match in matched_llvm_configs[j]:
                # print(config)
                assert match.check_condition(), "invalid match"
                print("===== llvm start =====")
                print("memory_updates:")
                for update in config.memory_updates:
                    print(f"  {update}")
                print("path conditions:")
                for path_condition in config.path_conditions:
                    print(f"  {path_condition}")
                print("matching substitution:", match.substitution)
                print("===== llvm end =====")
                # print("matching condition:", match.condition.simplify())

        print("### final configs")
        for config in final_dataflow_configs:
            print("===== dataflow start =====")
            print("memory_updates:")
            for update in config.memory_updates:
                print(f"  {update}")
            print("path conditions:")
            for path_condition in config.path_conditions:
                print(f"  {path_condition}")
            print("===== dataflow start =====")
        
        for config in final_llvm_configs:
            print("===== llvm start =====")
            print("memory_updates:")
            for update in config.memory_updates:
                print(f"  {update}")
            print("path conditions:")
            for path_condition in config.path_conditions:
                print(f"  {path_condition}")
            print("===== llvm end =====")


if __name__ == "__main__":
    main()

