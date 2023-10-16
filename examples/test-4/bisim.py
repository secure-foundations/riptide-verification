from typing import Dict

import json

import semantics.smt as smt
from semantics.matching import *

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def set_up_carry_gate(config: dataflow.Configuration, pe_id: int, name: str):
    dummy_permission = dataflow.PermissionVariable("dummy")
    pe = config.graph.vertices[pe_id]

    config.operator_states[pe_id].transition_to(dataflow.CarryOperator.loop)

    config.channel_states[pe.inputs[0].id].push(
        dataflow.PermissionedValue(
            smt.BVConst(1, dataflow.WORD_WIDTH),
            dummy_permission,
        ),
    )

    config.channel_states[pe.inputs[2].id].push(
        dataflow.PermissionedValue(
            smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_" + name + "_%d"),
            dummy_permission,
        ),
    )


def set_up_inv_gate(config: dataflow.Configuration, pe_id: int, name: str):
    dummy_permission = dataflow.PermissionVariable("dummy")
    pe = config.graph.vertices[pe_id]

    config.operator_states[pe_id].transition_to(dataflow.InvariantOperator.loop)
    config.operator_states[pe_id].value = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_" + name + "_%d")

    config.channel_states[pe.inputs[0].id].push(
        dataflow.PermissionedValue(
            smt.BVConst(1, dataflow.WORD_WIDTH),
            dummy_permission,
        ),
    )


def main():
    """
    See examples/bisim/test-4.png for operator and channel IDs in the dataflow graph
    """

    with open("examples/test-4/test-4.o2p") as dataflow_source:
        dfg = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
        # Set up initial config for the dataflow program
        free_vars: Dict[str, smt.SMTTerm] = {
            function_arg.variable_name: smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_param_{function_arg.variable_name}_%d")
            for function_arg in dfg.function_arguments
        }
        dataflow_init_config = dataflow.Configuration.get_initial_configuration(dfg, free_vars)

        dummy_permission = dataflow.PermissionVariable("dummy")

        dataflow_outer_config = dataflow_init_config.copy()
        dataflow_inner_config = dataflow_init_config.copy()

        # pop constant values
        dataflow_outer_config.channel_states[0].pop()
        dataflow_outer_config.channel_states[1].pop()
        dataflow_outer_config.channel_states[16].pop()
        dataflow_outer_config.channel_states[17].pop()
        dataflow_outer_config.channel_states[19].pop()
        dataflow_outer_config.channel_states[24].pop()

        # outer loop carry gates: 7, 9
        # outer loop inv gates: 1
        set_up_carry_gate(dataflow_outer_config, 7, "inc8")
        set_up_carry_gate(dataflow_outer_config, 9, "lso_alloc2_1_lcssa")
        set_up_inv_gate(dataflow_outer_config, 1, "smax")

        # pop constant values
        dataflow_inner_config.channel_states[0].pop()
        dataflow_inner_config.channel_states[1].pop()
        dataflow_inner_config.channel_states[16].pop()
        dataflow_inner_config.channel_states[17].pop()
        dataflow_inner_config.channel_states[19].pop()
        dataflow_inner_config.channel_states[24].pop()

        # inner loop carry gates: 18, 19, 20
        # inner loop inv gates: 10, 17
        set_up_carry_gate(dataflow_inner_config, 7, "inc8")
        set_up_carry_gate(dataflow_inner_config, 9, "lso_alloc2_1_lcssa")
        set_up_carry_gate(dataflow_inner_config, 18, "inc")
        set_up_carry_gate(dataflow_inner_config, 19, "add")
        set_up_carry_gate(dataflow_inner_config, 20, "4")
        set_up_inv_gate(dataflow_inner_config, 1, "smax")
        set_up_inv_gate(dataflow_inner_config, 10, "arrayidx")
        set_up_inv_gate(dataflow_inner_config, 17, "1")

        dataflow_inner_config.channel_states[32].push(
            dataflow.PermissionedValue(
                smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_i_0_%d"),
                dummy_permission,
            ),
        )
        dataflow_inner_config.channel_states[20].pop()
        dataflow_inner_config.channel_states[25].pop()

    with open("examples/test-4/test-4.lso.ll") as llvm_source:
        module = llvm.Parser.parse_module(llvm_source.read())
        function = tuple(module.functions.values())[0]
    
        llvm_init_config = llvm.Configuration.get_initial_configuration(module, function)

        llvm_outer_config = llvm.Configuration(
            module,
            function,
            current_block="for.cond",
            previous_block="for.cond.cleanup3",
            current_instr_counter=0,
            variables=OrderedDict([
                (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_A_%d")),
                (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_B_%d")),
                (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_param_len_%d")),
                (r"%smax", smt.FreshSymbol(smt.BVType(32), "llvm_var_smax_%d")),
                (r"%lso.alloc2.1.lcssa", smt.FreshSymbol(smt.BVType(32), "llvm_var_lso_alloc2_1_lcssa_%d")),
                (r"%inc8", smt.FreshSymbol(smt.BVType(32), "llvm_var_inc8_%d")),
            ]),
            path_conditions=[],
        )

        llvm_inner_config = llvm.Configuration(
            module,
            function,
            current_block="for.cond1",
            previous_block="for.body4",
            current_instr_counter=0,
            variables=OrderedDict([
                (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_A_%d")),
                (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_B_%d")),
                (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_param_len_%d")),
                (r"%smax", smt.FreshSymbol(smt.BVType(32), "llvm_var_smax_%d")), # TODO: should we leave this here?
                (r"%4", smt.FreshSymbol(smt.BVType(32), "llvm_var_4_%d")),
                (r"%add", smt.FreshSymbol(smt.BVType(32), "llvm_var_add_%d")),
                (r"%inc", smt.FreshSymbol(smt.BVType(32), "llvm_var_inc_%d")),
                (r"%i.0", smt.FreshSymbol(smt.BVType(32), "llvm_var_i_0_%d")),
                (r"%1", smt.FreshSymbol(smt.BVType(32), "llvm_var_1_%d")),
                (r"%arrayidx", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_var_arrayidx_%d")),
            ]),
            path_conditions=[],
        )

    dataflow_cut_points = [
        (dataflow_init_config, [
            0,
            6,
            1,
            9,
            7,
            12,
            16, 2, 15,
            11,
            5,

            # enters inner loop
            20,
            19,
            18,
            10,
            17,
            22,
            3, 14, 8, 21, 23,
            25,
            24,
            4,
            26,
        ]),
        (dataflow_outer_config, []),
        (dataflow_inner_config, []),
    ]

    llvm_cut_points = [
        llvm_init_config,
        llvm_outer_config,
        llvm_inner_config,
    ]

    for i, ((dataflow_cut_point, schedule), llvm_cut_point) in enumerate(zip(dataflow_cut_points, llvm_cut_points)):
        # if i != 0:
        #     continue

        print(f"##### trying cut point pair {i} #####")

        # configs matched to the invariant config

        # cut point index |-> configs matched to the cut point
        matched_dataflow_configs = { i: [] for i in range(len(dataflow_cut_points)) }
        matched_llvm_configs = { i: [] for i in range(len(llvm_cut_points)) }

        final_dataflow_configs = []
        final_llvm_configs = []

        # First try executing the dataflow cut point
        # queue = [(dataflow_cut_point.copy(), 0)]
        # while len(queue) != 0:
        #     config, num_steps = queue.pop(0)

        #     if num_steps >= len(schedule):
        #         print(config)

        #     assert num_steps < len(schedule), "dataflow execution goes beyond the existing schedule"
            
        #     # if num_steps == 16:
        #     #     print(config)

        #     next_operator = schedule[num_steps]

        #     if isinstance(next_operator, tuple):
        #         for op in next_operator:                
        #             results = config.copy().step_exhaust(op)
        #             if len(results) != 0:
        #                 break
        #     else:
        #         results = config.step_exhaust(next_operator)
   
        #     # print(f"[dataflow] === trying config after step {num_steps} (|next configs| = {len(results)})")

        #     for result in results:
        #         if isinstance(result, dataflow.NextConfiguration):
        #             # if num_steps == 10:
        #             #     print(result.config)

        #             # if num_steps == 15:
        #             #     print(result.config)
        #             # print(result.config.channel_states[25])

        #             for j, (dataflow_cut_point, _) in enumerate(dataflow_cut_points):
        #                 # print(f"[dataflow] trying to match with cut point {j}")
        #                 match = dataflow_cut_point.match(result.config)
        #                 if isinstance(match, MatchingSuccess):
        #                     print(f"[dataflow] !!! found a matched config at step {num_steps + 1} to cut point {j}")
        #                     # TODO: check match condition here
        #                     matched_dataflow_configs[j].append((result.config, match))
        #                     break
        #                 # print(f"[dataflow] matching failed: {match.reason}")
        #             else:
        #                 queue.append((result.config, num_steps + 1))

        #         elif isinstance(result, dataflow.StepException):
        #             assert False, f"got exception: {result.reason}"

        #     if len(results) == 0:
        #         print(f"[dataflow] !!! found a final config at step {num_steps}")
        #         final_dataflow_configs.append(config)

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

        print(f"### matched configs at cut point {j}")
        for j in range(len(dataflow_cut_points)):
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

        for j in range(len(llvm_cut_points)):
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

