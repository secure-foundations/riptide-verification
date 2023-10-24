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
            # print(f"stepped on pe {pe_id}")
            assert isinstance(results[0], dataflow.NextConfiguration)
            config = results[0].config
            updated = True
        
        elif len(results) > 1:
            # print(f"branched on pe {pe_id}")
            # branching, return immediately
            return results
        
        # otherwise if no step, continue
        
    if updated:
        return dataflow.NextConfiguration(config),

    return ()


def is_steer_inv(pe: dataflow.ProcessingElement) -> bool:
    return pe.operator == "CF_CFG_OP_STEER" or pe.operator == "CF_CFG_OP_INVARIANT"


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

    def branch(results: Tuple[dataflow.StepResult, ...], new_trace_counter: int):
        assert len(results) == 2

        # print("branching!!!")

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
                # print(correspondence_equations)
                assert False, f"bad branch on condition {results[1].config.path_conditions}; cannot determine which branch this path condition belongs to: {branch.config.path_conditions}"

        # print(len(first_branch), len(second_branch))

        assert len(first_branch) > 0
        assert len(second_branch) > 0

        # print("next counter", new_trace_counter)

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
            # print("skipping", position)
            continue
        pe_id_at_position = llvm_position_to_pe_id[position]
        results = run_dataflow_until_branch(config, (pe_id_at_position,))
        # print(f"firing trace index {trace_counter} ({position}, PE {pe_id_at_position})")
        if len(results) == 0:
            # print(config)
            assert False, f"trace index {trace_counter} ({position}, PE {pe_id_at_position}) not fireable"
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
            # print(config.path_conditions)
            decider_term = config.channel_states[channel_id].peek().term

            if check_implication(config.path_conditions, [
                smt.Equals(decider_term, smt.BVConst(1, dataflow.WORD_WIDTH))
            ]):
                decider_value = smt.BVConst(1, dataflow.WORD_WIDTH)
            elif check_implication(config.path_conditions, [
                smt.Equals(decider_term, smt.BVConst(0, dataflow.WORD_WIDTH))
            ]):
                decider_value = smt.BVConst(0, dataflow.WORD_WIDTH)
            else:
                assert False, f"cannot determine the decider value {decider_term} from path conditions {config.path_conditions}"

            init_config.channel_states[channel_id].push(
                dataflow.PermissionedValue(
                    decider_value,
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
                (r"%lso.alloc2.1", smt.FreshSymbol(smt.BVType(32), "llvm_var_lso_alloc2_1_%d")), # TODO: figure out how to generate this
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
        dataflow_init_config,
        None,
        None,
    ]

    llvm_cut_points = [
        llvm_init_config,
        llvm_outer_config,
        llvm_inner_config,
    ]

    # correspondence between LLVM variables and dataflow variables (excluding parameters)
    var_correspondence = [
        (),
        None, # to be inferred
        None, # to be inferred
    ]

    num_cut_points = len(llvm_cut_points)

    # matched_llvm_configs[i][j]: configs matched to i starting from j
    matched_llvm_configs = { i: { j: [] for j in range(num_cut_points) } for i in range(num_cut_points) }
    # final_llvm_configs[i]: final configs starting from i
    final_llvm_configs = { i: [] for i in range(num_cut_points) }

    matched_dataflow_configs = { i: { j: [] for j in range(num_cut_points) } for i in range(num_cut_points) }
    final_dataflow_configs = { i: [] for i in range(num_cut_points) }

    for i, llvm_cut_point in enumerate(llvm_cut_points):
        print(f"##### trying cut point pair {i} #####")
    
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

    # for function_arg in dfg.function_arguments

    def mirro_llvm_cut_point(cut_point_index: int):
        llvm_cut_point = llvm_cut_points[cut_point_index]
        dataflow_cut_point = dataflow_cut_points[cut_point_index]

        # Find all LLVM branches from the specified cut point
        llvm_branches = [
            branch
            for i in range(num_cut_points)
            for branch, _ in matched_llvm_configs[i][cut_point_index]
        ] + final_llvm_configs[cut_point_index]

        llvm_branch_to_cut_point_index = {
            id(branch): i
            for i in range(num_cut_points)
            for branch, _ in matched_llvm_configs[i][cut_point_index]
        }

        param_correspondence = tuple(
            (dataflow_free_vars[function_arg.variable_name], llvm_cut_point.variables["%" + function_arg.variable_name])
            for function_arg in dfg.function_arguments
        )

        mem_correspondence = (dataflow_cut_point.memory, llvm_cut_point.memory),

        matched_branches = run_dataflow_with_schedule(
            dataflow_cut_point.copy(),
            llvm_branches,
            param_correspondence + var_correspondence[cut_point_index] + mem_correspondence,
        )

        for llvm_branch, dataflow_branch in matched_branches:
            if id(llvm_branch) in llvm_branch_to_cut_point_index:
                target_cut_point_index = llvm_branch_to_cut_point_index[id(llvm_branch)]
                matched_dataflow_configs[target_cut_point_index][cut_point_index].append(dataflow_branch)

                # Infer the target dataflow cut point
                if dataflow_cut_points[target_cut_point_index] is None:
                    print(f"inferring dataflow cut point {target_cut_point_index} using a dataflow trace from cut point {cut_point_index}")
                    target_dataflow_cut_point, target_llvm_var_to_dataflow_var = generalize_dataflow_config(function, dataflow_init_config, dataflow_branch)
                    print(target_dataflow_cut_point)
                    target_var_correspondence = tuple(
                        (dataflow_smt_var, llvm_cut_points[target_cut_point_index].variables[llvm_var])
                        for llvm_var, dataflow_smt_vars in target_llvm_var_to_dataflow_var.items()
                        for dataflow_smt_var in dataflow_smt_vars
                    )
                    print(target_var_correspondence)

                    dataflow_cut_points[target_cut_point_index] = target_dataflow_cut_point
                    var_correspondence[target_cut_point_index] = target_var_correspondence

            else:
                final_dataflow_configs[cut_point_index].append(dataflow_branch)

    mirro_llvm_cut_point(0)
    mirro_llvm_cut_point(2)
    mirro_llvm_cut_point(1)

    for j in range(num_cut_points):
        for i in range(num_cut_points):
            print(f"[dataflow] {j} -> {i}:", len(matched_dataflow_configs[i][j]))

        print(f"[dataflow] {j} -> ⊥:", len(final_dataflow_configs[j]))

    # Actually check that matched_dataflow_configs matches the corresponding cut points
    for i in range(num_cut_points):
        for j in range(num_cut_points):
            for k, config in enumerate(matched_dataflow_configs[i][j]):
                match = dataflow_cut_points[i].match(config)
                if isinstance(match, MatchingSuccess):
                    assert match.check_condition(), "invalid match"
                    matched_dataflow_configs[i][j][k] = config, match
                else:
                    assert False, "match failure"

    for i in range(num_cut_points):
        for j in range(num_cut_points):
            print(f"### matched cut point {j} -> {i} ###")
            for config, match in matched_dataflow_configs[i][j]:
                print("===== dataflow start =====")
                print("memory_updates:")
                for update in config.memory_updates:
                    print(f"  {update}")
                print("path conditions:")
                for path_condition in config.path_conditions:
                    print(f"  {path_condition}")
                print("matching substitution:", match.substitution)
                print("===== dataflow end =====")

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

    for i in range(num_cut_points):
        print(f"### final configs from {i} ###")

        for config in final_dataflow_configs[i]:
            print("===== dataflow start =====")
            print("memory_updates:")
            for update in config.memory_updates:
                print(f"  {update}")
            print("path conditions:")
            for path_condition in config.path_conditions:
                print(f"  {path_condition}")
            print("===== dataflow end =====")

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

