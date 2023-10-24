from typing import Dict, Tuple, Iterable, List
from dataclasses import dataclass

import json

import semantics.smt as smt
from semantics.matching import *

import semantics.dataflow as dataflow
import semantics.llvm as llvm


dummy_permission = dataflow.PermissionVariable("dummy")


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

    # config.channel_states[pe.inputs[2].id].push(
    #     dataflow.PermissionedValue(
    #         smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_" + name + "_%d"),
    #         dummy_permission,
    #     ),
    # )


def set_up_inv_gate(config: dataflow.Configuration, pe_id: int, name: str):
    pe = config.graph.vertices[pe_id]

    config.operator_states[pe_id].transition_to(dataflow.InvariantOperator.loop)
    config.operator_states[pe_id].value = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_var_" + name + "_%d")

    config.channel_states[pe.inputs[0].id].push(
        dataflow.PermissionedValue(
            smt.BVConst(1, dataflow.WORD_WIDTH),
            dummy_permission,
        ),
    )


@dataclass
class LLVMExecutionBranch:
    config: llvm.Configuration
    trace: Tuple[Tuple[str, int], ...]


def run_dataflow_until_branch(
    config: dataflow.Configuration,
    pe_ids: Iterable[int],
) -> Tuple[dataflow.StepResult, ...]:
    """
    Run the specified PEs and return immediately if branching happens.
    If all given PEs are not fireable, return ()
    Otherwise return a single NextConfiguration
    """
    
    updated = False

    for pe_id in pe_ids:
        results = config.step_exhaust(pe_id)

        if len(results) == 1:
            print(f"stepped on pe {pe_id}")
            assert isinstance(results[0], dataflow.NextConfiguration)
            config = results[0].config
            updated = True
        
        elif len(results) > 1:
            print(f"branched on pe {pe_id}")
            # branching, return immediately
            return results
        
        # otherwise if no step, continue
        
    if updated:
        return dataflow.NextConfiguration(config),

    return ()


def is_steer_inv(pe: dataflow.ProcessingElement) -> bool:
    return pe.operator == "CF_CFG_OP_STEER" or pe.operator == "CF_CFG_OP_INVARIANT"


def run_dataflow_with_schedule(
    config: dataflow.Configuration,
    branches: Tuple[LLVMExecutionBranch, ...],
    correspondence: Tuple[Tuple[smt.SMTTerm, smt.SMTTerm], ...], # which SMT variables are equal
    trace_counter: int = 0,
) -> Tuple[Tuple[LLVMExecutionBranch, dataflow.Configuration], ...]:
    """
    Run a dataflow configuration with the same schedule as the LLVM branches
    Configs in branches should have disjoint path conditions

    Returning a tuple of final configurations (same length as branches)
    """

    assert len(branches) > 0

    # Build a mapping from llvm position to PE id
    llvm_position_to_pe_id: Dict[Tuple[str, int], int] = {
        pe.llvm_position: pe.id
        for pe in config.graph.vertices
        if pe.llvm_position is not None
    }

    # find all steer and invariant gates
    steer_inv_pe_ids = tuple(
        pe.id
        for pe in config.graph.vertices
        if is_steer_inv(pe)
    )

    def check_implication(a: Iterable[smt.SMTTerm], b: Iterable[smt.SMTTerm]) -> bool:
        # print("implication!!!")
        with smt.Solver(name="z3") as solver:
            for term in a:
                # print(term)
                solver.add_assertion(term)

            # for term in b:
                # print("=>", term)
            solver.add_assertion(smt.Not(smt.And(*b)))

            # unsat => implication valid
            result = not solver.solve()
            # print(result)

            return result

    def branch(results: Tuple[dataflow.StepResult, ...], new_trace_counter: int):
        assert len(results) == 2

        print("branching!!!")

        first_branch: List[LLVMExecutionBranch] = []
        second_branch: List[LLVMExecutionBranch] = []

        correspondence_equations = tuple(smt.Equals(a, b) for a, b in correspondence)

        for branch in branches:
            # Check if LLVM branch path condition /\ correspondence => dataflow branch path condition
            if check_implication(
                tuple(branch.config.path_conditions) + correspondence_equations,
                results[0].config.path_conditions,
            ):
                first_branch.append(branch)
            elif check_implication(
                tuple(branch.config.path_conditions) + correspondence_equations,
                results[1].config.path_conditions,
            ):
                second_branch.append(branch)
            else:
                assert False, f"bad branch on condition {results[0].config.path_conditions[-1]}"

        # print(len(first_branch), len(second_branch))

        assert len(first_branch) > 0
        assert len(second_branch) > 0

        print("next counter", new_trace_counter)

        return run_dataflow_with_schedule(results[0].config, first_branch, correspondence, new_trace_counter) + \
               run_dataflow_with_schedule(results[1].config, second_branch, correspondence, new_trace_counter)

    while True:
        # All llvm-position-labelled operators run as the schedule specifies
        # Other operators:
        # - Steer: always fire when available
        # - Inv: always fire when available (tentative, or run when the destination is fired)
        
        # Run steer/inv gates until stuck
        while True:
            results = run_dataflow_until_branch(config, steer_inv_pe_ids)
            if len(results) == 0:
                break
            elif len(results) > 1:
                return branch(results, trace_counter)
            else:
                config = results[0].config

        if len(branches) == 1 and trace_counter >= len(branches[0].trace):
            return (branches[0], config),

        assert len(set(branch.trace[trace_counter] for branch in branches)) == 1, \
               "early divergence"

        # Run the corresponding pe at trace_counter
        position = branches[0].trace[trace_counter]
        if position not in llvm_position_to_pe_id:
            # might have been coalesced into other PEs
            trace_counter += 1
            print("skipping", position)
            continue
        pe_id_at_position = llvm_position_to_pe_id[position]
        results = run_dataflow_until_branch(config, (pe_id_at_position,))
        if len(results) == 0:
            assert False, f"trace index {trace_counter} not fireable"
        elif len(results) > 1:
            return branch(results, trace_counter + 1)
        else:
            config = results[0].config

        # Run steer/inv gates again until stuck
        while True:
            results = run_dataflow_until_branch(config, steer_inv_pe_ids)
            if len(results) == 0:
                break
            elif len(results) > 1:
                return branch(results, trace_counter + 1)
            else:
                config = results[0].config

        # still no branching -- continue with the original trace
        trace_counter += 1


def find_non_steer_inv_producer(graph: dataflow.DataflowGraph, channel_id: int) -> Optional[int]:
    """
    Find the non-steer and non-inv producer of a channel
    """

    channel = graph.channels[channel_id]
    if channel.source is None:
        return None

    source_pe = graph.vertices[channel.source]

    if source_pe.operator == "CF_CFG_OP_STEER" or source_pe.operator == "CF_CFG_OP_INVARIANT":
        return find_non_steer_inv_producer(graph, source_pe.inputs[1].id)
    
    else:
        return source_pe.id


def generalize_dataflow_config(
    llvm_function: llvm.Function,
    init_config: dataflow.Configuration,
    config: dataflow.Configuration,
) -> Tuple[dataflow.Configuration, Dict[str, List[smt.SMTTerm]]]:
    """
    Generalize a dataflow config to a cut point.

    Returns the generalized template and a mapping from llvm variables to SMT variables used in channels
    """
    init_config = init_config.copy()

    llvm_correspondence: Dict[str, List[smt.SMTTerm]] = {}

    # Mirror the operator states
    for pe_id, operator in enumerate(config.operator_states):
        init_config.operator_states[pe_id].transition_to(operator.current_transition)

        if isinstance(operator, dataflow.InvariantOperator) and \
           operator.current_transition == dataflow.InvariantOperator.loop:
            actual_producer = find_non_steer_inv_producer(config.graph, config.graph.vertices[pe_id].inputs[1].id)

            if actual_producer is not None:
                producer_pe = config.graph.vertices[actual_producer]
                if producer_pe.llvm_position is not None:
                    block_name, instr_index = producer_pe.llvm_position
                    llvm_instr = llvm_function.blocks[block_name].instructions[instr_index]
                    defined_var = llvm_instr.get_defined_variable()
                    assert defined_var is not None, f"source pe {channel.source} corresponds to a non-definition {llvm_instr}"
                    sanitized_defined_var = defined_var[1:].replace('.', '_')
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_var_{sanitized_defined_var}_%d")
                    
                    if defined_var not in llvm_correspondence:
                        llvm_correspondence[defined_var] = []
                    llvm_correspondence[defined_var].append(fresh_var)
                else:
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_unknown_inv_channel_{channel.id}_%d")
            else:
                fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_unknown_inv_constant_{channel.id}_%d")

            init_config.operator_states[pe_id].value = fresh_var

    for channel_id, channel_state in enumerate(config.channel_states):
        if channel_state.hold_constant is not None:
            continue

        if channel_state.count() == 0:
            if init_config.channel_states[channel_id].ready():
                init_config.channel_states[channel_id].pop()
            continue

        # All channels should have at most one value
        assert channel_state.count() == 1, f"channel {channel_id} has more than one value"

        channel = config.graph.channels[channel_id]
        dest_pe = config.graph.vertices[channel.destination]

        assert init_config.channel_states[channel_id].count() <= 1
        if init_config.channel_states[channel_id].ready():
            init_config.channel_states[channel_id].pop()

        # If the destination is a carry, we need to simplify the decider condition to True
        if dest_pe.operator == "CF_CFG_OP_CARRY" and channel.destination_port == 0:
            # TODO: check if the channel value is always true under path condition
            init_config.channel_states[channel_id].push(
                dataflow.PermissionedValue(
                    smt.BVConst(1, dataflow.WORD_WIDTH),
                    dummy_permission,
                ),
            )

        else:
            # Generalize the value in the channel and assign an LLVM correspondence
            actual_producer = find_non_steer_inv_producer(config.graph, channel_id)

            if actual_producer is not None:
                producer_pe = config.graph.vertices[actual_producer]
                if producer_pe.llvm_position is not None:
                    block_name, instr_index = producer_pe.llvm_position
                    llvm_instr = llvm_function.blocks[block_name].instructions[instr_index]
                    defined_var = llvm_instr.get_defined_variable()
                    assert defined_var is not None, f"source pe {channel.source} corresponds to a non-definition {llvm_instr}"
                    sanitized_defined_var = defined_var[1:].replace('.', '_')
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_var_{sanitized_defined_var}_%d")

                    if defined_var not in llvm_correspondence:
                        llvm_correspondence[defined_var] = []
                    llvm_correspondence[defined_var].append(fresh_var)
                else:
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_unknown_channel_{channel.id}_%d")
            else:
                fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_unknown_constant_%d")

            init_config.channel_states[channel_id].push(
                dataflow.PermissionedValue(
                    fresh_var,
                    dummy_permission,
                ),
            )

    return init_config, llvm_correspondence


def main():
    """
    See examples/bisim/test-4.png for operator and channel IDs in the dataflow graph
    """

    with open("examples/test-4/test-4.o2p") as dataflow_source:
        dfg = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
        # Set up initial config for the dataflow program
        dataflow_free_vars: Dict[str, smt.SMTTerm] = {
            function_arg.variable_name: smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_param_{function_arg.variable_name}_%d")
            for function_arg in dfg.function_arguments
        }
        dataflow_init_config = dataflow.Configuration.get_initial_configuration(dfg, dataflow_free_vars)

        dataflow_outer_config = dataflow_init_config.copy()
        dataflow_inner_config = dataflow_init_config.copy()

        # pop constant values
        # dataflow_outer_config.channel_states[0].pop()
        # dataflow_outer_config.channel_states[1].pop()
        # dataflow_outer_config.channel_states[16].pop()
        # dataflow_outer_config.channel_states[17].pop()
        # dataflow_outer_config.channel_states[19].pop()
        # dataflow_outer_config.channel_states[24].pop()

        # outer loop carry gates: 7, 9
        # outer loop inv gates: 1
        set_up_carry_gate(dataflow_outer_config, 7, "inc8")
        set_up_carry_gate(dataflow_outer_config, 9, "lso_alloc2_1_lcssa")
        set_up_inv_gate(dataflow_outer_config, 1, "smax")

        # pop constant values
        # dataflow_inner_config.channel_states[0].pop()
        # dataflow_inner_config.channel_states[1].pop()
        # dataflow_inner_config.channel_states[16].pop()
        # dataflow_inner_config.channel_states[17].pop()
        # dataflow_inner_config.channel_states[19].pop()
        # dataflow_inner_config.channel_states[24].pop()

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

        # dataflow_inner_config.channel_states[32].push(
        #     dataflow.PermissionedValue(
        #         smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), "dataflow_i_0_%d"),
        #         dummy_permission,
        #     ),
        # )
        # dataflow_inner_config.channel_states[20].pop()
        # dataflow_inner_config.channel_states[25].pop()

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

    # dataflow_cut_points = [
    #     (dataflow_init_config, [
    #         0,
    #         6,
    #         1,
    #         9,
    #         7,
    #         12,
    #         16, 2, 15,
    #         11,
    #         5,

    #         # enters inner loop
    #         20,
    #         19,
    #         18,
    #         10,
    #         17,
    #         22,
    #         3, 14, 8, 21, 23,
    #         25,
    #         24,
    #         4,
    #         26,
    #     ]),
    #     (dataflow_outer_config, []),
    #     (dataflow_inner_config, []),
    # ]

    dataflow_cut_points = [
        dataflow_init_config,
        dataflow_outer_config,
        dataflow_inner_config,
    ]

    llvm_cut_points = [
        llvm_init_config,
        llvm_outer_config,
        llvm_inner_config,
    ]

    # matched_llvm_configs[i][j]: configs matched to i starting from j
    matched_llvm_configs = { i: { j: [] for j in range(len(llvm_cut_points)) } for i in range(len(llvm_cut_points)) }
    # final_llvm_configs[i]: final configs starting from i
    final_llvm_configs = { i: [] for i in range(len(llvm_cut_points)) }

    for i, llvm_cut_point in enumerate(llvm_cut_points):
        print(f"##### trying cut point pair {i} #####")

        # configs matched to the invariant config

        # cut point index |-> configs matched to the cut point
        # matched_dataflow_configs = { i: [] for i in range(len(dataflow_cut_points)) }
        # matched_llvm_configs = { i: [] for i in range(len(llvm_cut_points)) }

        # final_dataflow_configs = []
        # final_llvm_configs = []

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
        queue = [
            LLVMExecutionBranch(llvm_cut_point.copy(), ()),
        ]
       
        while len(queue) != 0:
            branch = queue.pop(0)
            current_position = (branch.config.current_block, branch.config.current_instr_counter)
            new_trace = branch.trace + (current_position,)
            results = branch.config.step()

            for result in results:
                if isinstance(result, llvm.NextConfiguration):
                    for j, llvm_cut_point in enumerate(llvm_cut_points):
                        match = llvm_cut_point.match(result.config)
                        if isinstance(match, MatchingSuccess):
                            print(f"[llvm] found a matched config to cut point {j}")
                            matched_llvm_configs[j][i].append((LLVMExecutionBranch(result.config, new_trace), match))
                            break
                    else:
                        queue.append(LLVMExecutionBranch(result.config, new_trace))

                elif isinstance(result, llvm.FunctionReturn):
                    print(f"[llvm] found a final config")
                    final_llvm_configs[i].append(LLVMExecutionBranch(branch.config, new_trace))

                else:
                    assert False, f"unsupported result {result}"

    # for i in range(1, len(llvm_cut_points)):
    #     print(f"### [dataflow] generating cut point {i} ###")
    #     # TODO: here we assume the carry and invariant gate states are already set

    #     for channel in dfg.channels:
    #         # hold constant channel should always have the same state
    #         if channel.hold and channel.constant is not None:
    #             continue
            
    #         dest_operator = dfg.vertices[channel.destination]

    #         if channel.constant is not None:
    #             assert dest_operator.llvm_position is not None

    #             consumed = None
    #             # check if the dest operator is ever executed in any cut point matches TO i
    #             for j in range(len(llvm_cut_points)):
    #                 for branch, match in matched_llvm_configs[i][j]:
    #                     if dest_operator.llvm_position in branch.trace:
    #                         set_consumed = True
    #                     else:
    #                         set_consumed = False

    #                     if consumed is None:
    #                         consumed = set_consumed
    #                     else:
    #                         assert consumed == set_consumed, f"inconsistent constant usage for channel {channel.id}"

    #             if consumed:
    #                 # pop constant if all matched config at cut point i consumes this constant
    #                 dataflow_cut_points[i].channel_states[channel.id].pop()
    #                 print(f"constant channel {channel.id} popped")
    #             else:
    #                 print(f"constant channel {channel.id} not popped")
                
    #             continue

    #         # otherwise there should always be a source operator
    #         source_operator = dfg.vertices[channel.source]

    #         # non-constant channels should be empty in the init state,
    #         # otherwise this channel is already prepared in some preprocessing stage
    #         # so we don't overwrite that
    #         if dataflow_cut_points[i].channel_states[channel.id].ready():
    #             continue

    #         if dest_operator.llvm_position is not None:
    #             # find the true source operator (skipping any steer gates and inv gates that are not ready)
                
    #             while True:
    #                 if source_operator.operator == "CF_CFG_OP_STEER":
    #                     prev_source = source_operator.inputs[1].source
    #                     assert prev_source is not None
    #                     source_operator = dfg.vertices[prev_source]

    #                 elif source_operator.operator == "CF_CFG_OP_INVARIANT":
    #                     prev_source = source_operator.inputs[1].source
    #                     assert prev_source is not None
    #                     source_operator = dfg.vertices[prev_source]

    #                 else:
    #                     break

    #             assert source_operator.llvm_position is not None

    #             # check if the source operator (modulo any steer/inv gates) is ever executed in any cut point matches to i
    #             source_have_executed = None
    #             for j in range(len(llvm_cut_points)):
    #                 for branch, match in matched_llvm_configs[j][i]:
    #                     if dest_operator.llvm_position in branch.trace:
    #                         set_source_have_executed = True
    #                     else:
    #                         set_source_have_executed = False

    #                     if source_have_executed is None:
    #                         source_have_executed = set_source_have_executed
    #                     else:
    #                         assert source_have_executed == set_source_have_executed, \
    #                                f"inconsistent source operator usage for channel {channel.id}"
            
    #             # check if the dest operator is ever executed in any cut point matches STARTING FROM i
    #             dest_will_execute = None
    #             for j in range(len(llvm_cut_points)):
    #                 for branch, match in matched_llvm_configs[j][i]:
    #                     if dest_operator.llvm_position in branch.trace:
    #                         set_dest_will_execute = True
    #                     else:
    #                         set_dest_will_execute = False

    #                     if dest_will_execute is None:
    #                         dest_will_execute = set_dest_will_execute
    #                     else:
    #                         assert dest_will_execute == set_dest_will_execute, \
    #                                f"inconsistent dest operator usage for channel {channel.id}"
                            
    #             print(f"channel {channel.id} usage:", source_have_executed, dest_will_execute)

    llvm_branches_from_init = [ branch for i in range(len(llvm_cut_points)) for branch, match in matched_llvm_configs[i][0] ] +\
                              final_llvm_configs[0]

    param_correspondence = (
        (dataflow_free_vars["A"], llvm_init_config.variables["%A"]),
        (dataflow_free_vars["B"], llvm_init_config.variables["%B"]),
        (dataflow_free_vars["len"], llvm_init_config.variables["%len"]),
    )

    dataflow_branches = run_dataflow_with_schedule(
        dataflow_init_config.copy(),
        llvm_branches_from_init,
        param_correspondence,
    )
    
    for llvm_branch, dataflow_config in dataflow_branches:
        print(llvm_branch.config)
        print(llvm_branch.trace)
        print(dataflow_config)

    dataflow_cut_point_2, correspondence = generalize_dataflow_config(function, dataflow_init_config, dataflow_branches[1][1])

    print(dataflow_cut_point_2)
    print(correspondence)

    # same but for cut point
    llvm_branches_from_init = [ branch for i in range(len(llvm_cut_points)) for branch, match in matched_llvm_configs[i][2] ] +\
                              final_llvm_configs[2]

    # (r"%A", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_A_%d")),
    # (r"%B", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_param_B_%d")),
    # (r"%len", smt.FreshSymbol(smt.BVType(32), "llvm_param_len_%d")),
    # (r"%smax", smt.FreshSymbol(smt.BVType(32), "llvm_var_smax_%d")), # TODO: should we leave this here?
    # (r"%4", smt.FreshSymbol(smt.BVType(32), "llvm_var_4_%d")),
    # (r"%add", smt.FreshSymbol(smt.BVType(32), "llvm_var_add_%d")),
    # (r"%inc", smt.FreshSymbol(smt.BVType(32), "llvm_var_inc_%d")),
    # (r"%i.0", smt.FreshSymbol(smt.BVType(32), "llvm_var_i_0_%d")),
    # (r"%1", smt.FreshSymbol(smt.BVType(32), "llvm_var_1_%d")),
    # (r"%arrayidx", smt.FreshSymbol(smt.BVType(llvm.WORD_WIDTH), "llvm_var_arrayidx_%d")),

    param_correspondence = (
        (dataflow_free_vars["A"], llvm_cut_points[2].variables["%A"]),
        (dataflow_free_vars["B"], llvm_cut_points[2].variables["%B"]),
        (dataflow_free_vars["len"], llvm_cut_points[2].variables["%len"]),
    )
    correspondence = tuple((dataflow_smt_var, llvm_cut_points[2].variables[llvm_var]) for llvm_var, dataflow_smt_vars in correspondence.items() for dataflow_smt_var in dataflow_smt_vars)

    dataflow_branches = run_dataflow_with_schedule(
        dataflow_cut_point_2.copy(),
        llvm_branches_from_init,
        param_correspondence + correspondence,
    )

    dataflow_cut_point_1, correspondence = generalize_dataflow_config(function, dataflow_init_config, dataflow_branches[0][1])

    print(dataflow_cut_point_1)
    print(correspondence)

    dataflow_cut_points = (
        dataflow_init_config,
        dataflow_cut_point_1,
        dataflow_cut_point_2,
    )

    return

    for i in range(len(llvm_cut_points)):
        for j in range(len(llvm_cut_points)):
            print(f"### [llvm] matched cut point {j} -> {i} ###")
            for branch, match in matched_llvm_configs[i][j]:
                # print(config)
                assert match.check_condition(), "invalid match"
                print("===== llvm start =====")
                print("memory_updates:")
                for update in branch.config.memory_updates:
                    print(f"  {update}")
                print("path conditions:")
                for path_condition in branch.config.path_conditions:
                    print(f"  {path_condition}")
                print("matching substitution:", match.substitution)
                print("trace:", branch.trace)
                print("===== llvm end =====")
                # print("matching condition:", match.condition.simplify())

    for i in range(len(llvm_cut_points)):
        print(f"### [llvm] final configs from {i} ###")
        for branch in final_llvm_configs[i]:
            print("===== llvm start =====")
            print("memory_updates:")
            for update in branch.config.memory_updates:
                print(f"  {update}")
            print("path conditions:")
            for path_condition in branch.config.path_conditions:
                print(f"  {path_condition}")
            print("trace:", branch.trace)
            print("===== llvm end =====")


if __name__ == "__main__":
    main()

