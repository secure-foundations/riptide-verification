from __future__ import annotations

from typing import Tuple, Iterable, Optional, List, OrderedDict, Dict
from dataclasses import dataclass
from collections import OrderedDict

import sys

import semantics.smt as smt
from semantics.matching import *

import semantics.dataflow as dataflow
import semantics.llvm as llvm


@dataclass
class LoopHeaderHint:
    block_name: str
    prev_block: str
    # live_vars: Tuple[str, ...] # live variables (excluding parameters)
    lcssa_vars: Tuple[Tuple[str, str], ...] # pairs of (original var, lcssa var)


@dataclass
class LLVMBranch:
    config: llvm.Configuration
    match_result: Optional[MatchingSuccess] # None for final configuration
    trace: Tuple[Tuple[str, int], ...] # pairs of (block name, instr index)
    from_cut_point: int
    to_cut_point: Optional[int] # None for final configuration


@dataclass
class DataflowBranch:
    config: dataflow.Configuration
    match_result: Optional[MatchingSuccess] # None for final configuration
    llvm_branch: LLVMBranch # corresponding llvm branch


@dataclass
class Correspondence:
    param_correspondence: Tuple[Tuple[smt.SMTTerm, smt.SMTTerm], ...]
    mem_correspondence: Tuple[smt.SMTTerm, smt.SMTTerm]
    var_correspondence: Tuple[Tuple[smt.SMTTerm, smt.SMTTerm], ...]

    def get_all(self) -> Tuple[Tuple[smt.SMTTerm, smt.SMTTerm], ...]:
        return (
            (self.mem_correspondence[0], self.mem_correspondence[1]),
            *((left, right) for left, right in self.param_correspondence),
            *((left, right) for left, right in self.var_correspondence),
        )

    def __str__(self) -> str:
        return " /\\ ".join(f"{left} = {right}" for left, right in self.get_all())

    def to_smt_terms(self) -> Tuple[smt.SMTTerm, ...]:
        return tuple(smt.Equals(left, right) for left, right in self.get_all())

    def to_smt_term(self) -> smt.SMTTerm:
        return smt.And(*self.to_smt_terms())
    
    def get_matching_obligations(self, dataflow_match: MatchingSuccess, llvm_match: MatchingSuccess) -> Tuple[smt.SMTTerm, ...]:
        """
        Get the equalities after substitutions in dataflow_match and llvm_match
        """

        return tuple(
            smt.Equals(
                left.substitute(dataflow_match.substitution),
                right.substitute(llvm_match.substitution),
            )
            for left, right in self.get_all()
        )


class SimulationChecker:
    @staticmethod
    def sanitize_llvm_name(name: str) -> str:
        """
        Sanitize an identifier in LLVM to something usable in SMT
        """
        if name.startswith("%") or name.startswith("@"):
            name = name[1:]
        name = name.replace(".", "_")
        return name
    
    @staticmethod
    def llvm_type_to_smt_type(type: llvm.Type) -> smt.SMTTerm:
        if isinstance(type, llvm.IntegerType):
            return smt.BVType(type.bit_width)
        
        elif isinstance(type, llvm.PointerType):
            return smt.BVType(llvm.WORD_WIDTH)
        
        assert False, f"unsupported llvm type {type}"

    def __init__(
        self,
        dataflow_graph: dataflow.DataflowGraph,
        llvm_function: llvm.Function,
        loop_header_hints: Iterable[LoopHeaderHint],
        debug: bool = True,
    ):
        self.debug = debug
        self.dataflow_graph = dataflow_graph
        self.llvm_function = llvm_function
        self.loop_header_hints = tuple(loop_header_hints)

        # Build a mapping from llvm position to PE id
        self.llvm_position_to_pe_id: Dict[Tuple[str, int], int] = {
            pe.llvm_position: pe.id
            for pe in dataflow_graph.vertices
            if pe.llvm_position is not None
        }

        # Find all steer and inv gates
        self.steer_inv_pe_ids = tuple(
            pe.id
            for pe in dataflow_graph.vertices
            if pe.operator == "CF_CFG_OP_STEER" or pe.operator == "CF_CFG_OP_INVARIANT"
        )

        # Find all carry gates
        self.carry_pe_ids = tuple(
            pe.id
            for pe in dataflow_graph.vertices
            if pe.operator == "CF_CFG_OP_CARRY"
        )

        # Set up initial configs
        assert len(self.llvm_function.parameters) == len(self.dataflow_graph.function_arguments)

        self.dataflow_params: OrderedDict[str, smt.SMTTerm] = OrderedDict(
            (function_arg.variable_name, smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_param_{function_arg.variable_name}_%d"))
            for function_arg in self.dataflow_graph.function_arguments
        )

        dataflow_init_config = dataflow.Configuration.get_initial_configuration(self.dataflow_graph, self.dataflow_params)
        llvm_init_config = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)

        self.num_cut_points = 1 + len(loop_header_hints)
        self.dataflow_cut_points: List[Optional[dataflow.Configuration]] = [ dataflow_init_config ]
        self.dataflow_cut_points_executed: List[bool] = [ False ] * self.num_cut_points
        self.llvm_cut_points: List[llvm.Configuration] = [ llvm_init_config ]

        self.correspondence: List[Optional[Correspondence]] = [ self.get_init_correspondence(dataflow_init_config, llvm_init_config) ]

        # self.matched_dataflow_branches[i][j] = branches from cut point j matched to cut point i
        self.matched_dataflow_branches: Tuple[Tuple[List[DataflowBranch], ...], ...] = tuple(
            tuple([] for _ in range(self.num_cut_points))
            for _ in range(self.num_cut_points)
        )
        self.matched_llvm_branches: Tuple[Tuple[LLVMBranch, ...], ...] = tuple(
            tuple([] for _ in range(self.num_cut_points))
            for _ in range(self.num_cut_points)
        )
        self.final_dataflow_branches: Tuple[List[DataflowBranch], ...] = tuple([] for _ in range(self.num_cut_points))
        self.final_llvm_branches: Tuple[List[LLVMBranch], ...] = tuple([] for _ in range(self.num_cut_points))

        # For each loop header, generate LLVM cut point and a placeholder for dataflow cut point
        for header_info in loop_header_hints:
            self.dataflow_cut_points.append(None)
            llvm_cut_point = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)
            llvm_cut_point.current_block = header_info.block_name
            llvm_cut_point.previous_block = header_info.prev_block
            # llvm_cut_point = llvm.Configuration(
            #     self.llvm_function.module,
            #     self.llvm_function,
            #     current_block=header_info.block_name,
            #     previous_block=header_info.prev_block,
            #     current_instr_counter=0,
            #     variables=OrderedDict([
            #         # Free variables for parameters
            #         *(
            #             (param.name, smt.FreshSymbol(
            #                 SimulationChecker.llvm_type_to_smt_type(param.type),
            #                 f"llvm_param_{SimulationChecker.sanitize_llvm_name(param.name)}_%d",
            #             ))
            #             for param in self.llvm_function.parameters.values()
            #         ),

            #         # Free variables for live variables
            #         *(
            #             (live_var_name, smt.FreshSymbol(
            #                 SimulationChecker.llvm_type_to_smt_type(self.llvm_function.definitions[live_var_name].get_type()),
            #                 f"llvm_var_{SimulationChecker.sanitize_llvm_name(live_var_name)}_%d",
            #             ))
            #             for live_var_name in header_info.live_vars
            #         ),
            #     ]),
            #     path_conditions=[],
            # )

            for original_var_name, lcssa_var_name in header_info.lcssa_vars:
                llvm_cut_point.set_variable(original_var_name, llvm_cut_point.get_variable(lcssa_var_name))

            self.llvm_cut_points.append(llvm_cut_point)
            self.correspondence.append(None)

    def debug_common(self, *args, **kwargs):
        if self.debug:
            print("[common]", *args, **kwargs, file=sys.stderr, flush=True)

    def debug_dataflow(self, *args, **kwargs):
        if self.debug:
            print("[dataflow]", *args, **kwargs, file=sys.stderr, flush=True)

    def debug_llvm(self, *args, **kwargs):
        if self.debug:
            print("[llvm]", *args, **kwargs, file=sys.stderr, flush=True)

    def get_pe_id_from_llvm_position(self, position: Tuple[str, int]) -> Optional[int]:
        return self.llvm_position_to_pe_id.get(position)

    def match_llvm_branches(self) -> None:
        """
        Run llvm cut points and match the branches against all cut points (or final configs)
        This will fill up self.matched_llvm_branches and self.final_llvm_branches
        """
        for i, llvm_cut_point in enumerate(self.llvm_cut_points):
            self.debug_llvm(f"running cut point {i}")

            # Queue of pairs of (llvm config, trace)
            queue: List[Tuple[llvm.Configuration, Tuple[Tuple[str, int], ...]]] = [
                (llvm_cut_point.copy(), ()),
            ]
        
            while len(queue) != 0:
                config, trace = queue.pop(0)
                current_position = (config.current_block, config.current_instr_counter)
                new_trace = trace + (current_position,)
                results = config.step()

                for result in results:
                    if isinstance(result, llvm.NextConfiguration):
                        for j, llvm_cut_point in enumerate(self.llvm_cut_points):
                            match_result = llvm_cut_point.match(result.config)
                            if isinstance(match_result, MatchingSuccess):
                                assert match_result.check_condition(), f"invalid match at cut point {j}"
                                self.debug_llvm(f"found a matching config to cut point {j}")
                                self.matched_llvm_branches[j][i].append(LLVMBranch(result.config, match_result, new_trace, i, j))
                                break
                        else:
                            # continue execution
                            queue.append((result.config, new_trace))

                    elif isinstance(result, llvm.FunctionReturn):
                        self.debug_llvm("found a final config")
                        self.final_llvm_branches[i].append(LLVMBranch(config, None, new_trace, i, None))

                    else:
                        assert False, f"unsupported llvm execution result {result}"

    def bin_llvm_branches_by_dataflow_branching(
        self,
        left: dataflow.Configuration,
        right: dataflow.Configuration,
        llvm_branches: Tuple[LLVMBranch, ...],
        correspondence: Correspondence,
    ) -> Tuple[Tuple[LLVMBranch, ...], Tuple[LLVMBranch, ...]]:
        left_branches: List[LLVMBranch] = []
        right_branches: List[LLVMBranch] = []

        correspondence_smt = correspondence.to_smt_terms()

        for branch in llvm_branches:
            branch_conditions = tuple(branch.config.path_conditions) + correspondence_smt

            # Check if LLVM branch path condition /\ correspondence => dataflow branch path condition
            if smt.check_implication(branch_conditions, left.path_conditions):
                left_branches.append(branch)
            elif smt.check_implication(branch_conditions, right.path_conditions):
                right_branches.append(branch)
            else:
                blame_left = smt.find_implication_blame(branch_conditions, left.path_conditions)
                blame_right = smt.find_implication_blame(branch_conditions, right.path_conditions)
                self.debug_common(f"correspondence: {correspondence}")
                self.debug_common(f"llvm branch path conditions: {branch.config.path_conditions}")
                self.debug_common(f"left blame:", blame_left)
                self.debug_common(f"right blame:", blame_right)
                assert False, f"failed to categorize a llvm branch into neither dataflow branches"

        return left_branches, right_branches

    def run_dataflow_with_llvm_branches(
        self,
        config: dataflow.Configuration,
        llvm_branches: Tuple[LLVMBranch, ...],
        correspondence: Correspondence,
        trace_counter: int = 0,
    ) -> Tuple[DataflowBranch, ...]:
        """
        Run a dataflow configuration with the same schedule as the LLVM branches
        Configs in branches should have disjoint path conditions
        Each LLVM branch should correspond to exactly one dataflow branch

        Returning a tuple of dataflow branches (same length as branches)
        """

        assert len(llvm_branches) > 0, f"dataflow config {config} has no matching llvm branches"

        def run_misc_operators() -> Optional[Tuple[DataflowBranch, ...]]:
            nonlocal config

            while True:
                changed = False

                # Run steer/inv gates until stuck
                results = config.step_until_branch(self.steer_inv_pe_ids)
                if len(results) == 1:
                    changed = True
                    config = results[0].config
                elif len(results) > 1:
                    return branch(results)

                # Step on all carry gates with a decider value
                results = config.step_until_branch(
                    (
                        pe_id
                        for pe_id in self.carry_pe_ids
                        if config.operator_states[pe_id].current_transition == dataflow.CarryOperator.loop
                    ),
                    exhaust=False,
                )
                if len(results) == 1:
                    changed = True
                    config = results[0].config
                elif len(results) > 1:
                    print("branch in carry")
                    return branch(results)
                
                if not changed:
                    break

            return None

        def branch(results: Tuple[dataflow.StepResult, ...]) -> Tuple[DataflowBranch, ...]:
            assert len(results) == 2 and \
                   isinstance(results[0], dataflow.NextConfiguration) and \
                   isinstance(results[1], dataflow.NextConfiguration), "irregular branching"
            left_branches, right_branches = self.bin_llvm_branches_by_dataflow_branching(
                results[0].config, results[1].config, llvm_branches, correspondence,
            )
            return self.run_dataflow_with_llvm_branches(results[0].config, left_branches, correspondence, trace_counter) + \
                   self.run_dataflow_with_llvm_branches(results[1].config, right_branches, correspondence, trace_counter)

        while True:
            # All llvm-position-labelled operators run as the schedule specifies
            # Other operators:
            # - Steer: always fire when available
            # - Inv: always fire when available (tentative, or run when the destination is fired)
        
            dataflow_branches = run_misc_operators()
            if dataflow_branches:
                return dataflow_branches

            # Base case: exactly one llvm branch
            if len(llvm_branches) == 1 and trace_counter >= len(llvm_branches[0].trace):
                return DataflowBranch(config, None, llvm_branches[0]),

            possible_llvm_positions = set(branch.trace[trace_counter] for branch in llvm_branches)
            assert len(possible_llvm_positions) == 1, \
                   f"dataflow did not branch but llvm branches at trace counter {trace_counter}: {possible_llvm_positions}"

            # Run the corresponding pe at trace_counter
            position = llvm_branches[0].trace[trace_counter]
            trace_counter += 1

            pe_id = self.get_pe_id_from_llvm_position(position)
            if pe_id is None:
                # This instruction might have been coalesced into other PEs
                continue
            
            results = config.step_until_branch((pe_id,))
            if len(results) == 1:
                config = results[0].config
            elif len(results) > 1:
                return branch(results)
            else:
                self.debug_common(config)
                assert False, f"PE {pe_id} corresponding to llvm instruction {position} not ready when scheduled to fire"

            dataflow_branches = run_misc_operators()
            if dataflow_branches:
                return dataflow_branches

    def find_non_steer_inv_producer(self, channel_id: int) -> Optional[int]:
        """
        Find the non-steer and non-inv producer of a channel
        """

        channel = self.dataflow_graph.channels[channel_id]
        if channel.source is None:
            return None

        source_pe = self.dataflow_graph.vertices[channel.source]

        if source_pe.operator == "CF_CFG_OP_STEER" or source_pe.operator == "CF_CFG_OP_INVARIANT":
            return self.find_non_steer_inv_producer(source_pe.inputs[1].id)
        else:
            return source_pe.id
    
    def find_non_steer_inv_producer_llvm_var(self, channel_id: int) -> Optional[str]:
        """
        Find the llvm variable name of the actual producer of a value in channel_id
        """
        producer_pe_id = self.find_non_steer_inv_producer(channel_id)

        if producer_pe_id is not None:
            producer_pe = self.dataflow_graph.vertices[producer_pe_id]
            if producer_pe.llvm_position is not None:
                # Find the corresponding llvm var
                block_name, instr_index = producer_pe.llvm_position
                llvm_instr = self.llvm_function.blocks[block_name].instructions[instr_index]
                defined_var = llvm_instr.get_defined_variable()
                return defined_var

        return None
    
    def check_branch_bisimulation_obligation(self, dataflow_branch: DataflowBranch):
        llvm_branch = dataflow_branch.llvm_branch
        self.debug_common(f"checking bisimulation obligations for a branch from cut point {llvm_branch.from_cut_point} to {llvm_branch.to_cut_point or '⊥'}")
    
        source_correspondence = self.correspondence[llvm_branch.from_cut_point]
        source_correspondence_smt = source_correspondence.to_smt_terms()

        obligations = [
            # Path condition equivalence
            smt.Iff(
                smt.And(*dataflow_branch.config.path_conditions),
                smt.And(*llvm_branch.config.path_conditions),
            )
        ]

        if llvm_branch.to_cut_point is not None:
            target_correspondence = self.correspondence[llvm_branch.to_cut_point]
            assert dataflow_branch.match_result is not None
            assert llvm_branch.match_result is not None
            obligations.extend(target_correspondence.get_matching_obligations(dataflow_branch.match_result, llvm_branch.match_result))
        else:
            # Correspondence at the final state is simply the memory equality
            obligations.append(smt.Equals(dataflow_branch.config.memory, llvm_branch.config.memory))

        if not smt.check_implication(source_correspondence_smt, obligations):
            blame = smt.find_implication_blame(source_correspondence_smt, obligations)
            self.debug_common("knows:", source_correspondence)
            self.debug_common("blame:", blame)
            assert False, f"a branch from {llvm_branch.from_cut_point} to {llvm_branch.to_cut_point or '⊥'} fails the bisimulation obligations"


    def check_bisimulation(self):
        """
        Check if all the matched dataflow/llvm branches satisfy cut point correspondence (thus establishing a bisimulation)
        """
        for j in range(self.num_cut_points):
            for i in range(self.num_cut_points):
                for dataflow_branch in self.matched_dataflow_branches[i][j]:
                    self.check_branch_bisimulation_obligation(dataflow_branch)

        for i in range(self.num_cut_points):
            for dataflow_branch in self.final_dataflow_branches[i]:
                self.check_branch_bisimulation_obligation(dataflow_branch)

    def check_dataflow_matches(self):
        """
        Check if the branches in self.matched_dataflow_branches actually match
        their corresponding cut points. If so set branch.match_result
        """

        for i in range(self.num_cut_points):
            target_dataflow_cut_point = self.dataflow_cut_points[i]

            for j in range(self.num_cut_points):
                for dataflow_branch in self.matched_dataflow_branches[i][j]:
                    self.debug_dataflow(f"checking a matched branch from cut point {j} to {i}")
                    match_result = target_dataflow_cut_point.match(dataflow_branch.config)
                    assert isinstance(match_result, MatchingSuccess), \
                           f"failed to match an expected dataflow branch from cut point {j} to {i}: {match_result.reason}"
                    assert match_result.check_condition(), "unexpected matching failure"
                    dataflow_branch.match_result = match_result

    def generate_dataflow_cut_points(self):
        while True:
            # find the next ready but not executed cut point
            for i in range(self.num_cut_points):
                if self.dataflow_cut_points[i] is not None and \
                   not self.dataflow_cut_points_executed[i]:
                    # Mirror the LLVM execution in the dataflow cut point
                    self.mirror_llvm_cut_point(i)
                    break
            else:
                # check if any cut point is useless
                for i in range(self.num_cut_points):
                    if self.dataflow_cut_points[i] is None:
                        assert False, f"unreachable cut point {i}"
                break

    def mirror_llvm_cut_point(self, cut_point_index: int):
        """
        Mirror the execution of LLVM cut point on the corresponding dataflow cut point

        This will fill in self.dataflow_cut_points at which point the specified cut point reaches
        """

        assert not self.dataflow_cut_points_executed[cut_point_index]
        self.dataflow_cut_points_executed[cut_point_index] = True
 
        dataflow_cut_point = self.dataflow_cut_points[cut_point_index]
        correspondence = self.correspondence[cut_point_index]

        assert dataflow_cut_point is not None and correspondence is not None

        # Find all LLVM branches from the specified cut point
        llvm_branches = [
            branch
            for i in range(self.num_cut_points)
            for branch in self.matched_llvm_branches[i][cut_point_index]
        ] + self.final_llvm_branches[cut_point_index]

        dataflow_branches = self.run_dataflow_with_llvm_branches(dataflow_cut_point.copy(), llvm_branches, correspondence)

        for dataflow_branch in dataflow_branches:
            target_cut_point = dataflow_branch.llvm_branch.to_cut_point
            if target_cut_point is not None:
                self.matched_dataflow_branches[target_cut_point][cut_point_index].append(dataflow_branch)

                # Infer the target dataflow cut point
                if self.dataflow_cut_points[target_cut_point] is None:
                    self.debug_dataflow(f"inferring dataflow cut point {target_cut_point} using a dataflow trace from cut point {cut_point_index}")
                    target_dataflow_cut_point, target_correspondence = self.generalize_dataflow_branch_to_cut_point(dataflow_branch)
                    self.debug_dataflow(f"inferred dataflow cut point {target_cut_point}\n{target_dataflow_cut_point}")
                    self.debug_dataflow(f"inferred correspondence at cut point {target_cut_point}\n{target_correspondence}")

                    self.dataflow_cut_points[target_cut_point] = target_dataflow_cut_point
                    self.correspondence[target_cut_point] = target_correspondence

                else:
                    # TODO: check if the generalized config matches anyway?
                    ...
            else:
                self.final_dataflow_branches[cut_point_index].append(dataflow_branch)

    def generalize_dataflow_branch_to_cut_point(self, branch: DataflowBranch) -> Tuple[dataflow.Configuration, Correspondence]:
        """
        Generalize a dataflow config to a cut point.

        Returns the generalized template and a correspondence
        """
        cut_point = self.dataflow_cut_points[0].copy()

        # TODO
        dummy_permission = dataflow.PermissionVariable("dummy")

        # Mapping from llvm var name |-> generalized dataflow variables corresponding to it
        llvm_var_correspondence: OrderedDict[str, List[smt.SMTTerm]] = OrderedDict()

        llvm_cut_point = self.llvm_cut_points[branch.llvm_branch.to_cut_point]

        # Mirror the operator states
        for pe_id, operator in enumerate(branch.config.operator_states):
            cut_point.operator_states[pe_id].transition_to(operator.current_transition)

            if isinstance(operator, dataflow.InvariantOperator) and \
               operator.current_transition == dataflow.InvariantOperator.loop:
                input_channel_id = self.dataflow_graph.vertices[pe_id].inputs[1].id
                producer_llvm_var = self.find_non_steer_inv_producer_llvm_var(input_channel_id)

                if producer_llvm_var is not None:
                    sanitized_var = SimulationChecker.sanitize_llvm_name(producer_llvm_var)
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_var_{sanitized_var}_%d")
                    cut_point.operator_states[pe_id].value = fresh_var

                    assert llvm_cut_point.has_variable(producer_llvm_var),\
                           f"corresponding llvm var {producer_llvm_var} of an invariant value at PE {pe_id} is not defined at the llvm cut point {branch.llvm_branch.to_cut_point}"
                    if producer_llvm_var not in llvm_var_correspondence:
                        llvm_var_correspondence[producer_llvm_var] = []
                    llvm_var_correspondence[producer_llvm_var].append(fresh_var)
                else:
                    assert False, f"cannot find an llvm variable corresponding to an invariant value in PE {pe_id}"

        # Generalize the channel states
        for channel_id, channel_state in enumerate(branch.config.channel_states):
            if channel_state.hold_constant is not None:
                continue

            if channel_state.count() == 0:
                if cut_point.channel_states[channel_id].ready():
                    cut_point.channel_states[channel_id].pop()
                continue

            # All channels should have at most one value
            assert channel_state.count() == 1, f"channel {channel_id} has more than one value"

            channel = branch.config.graph.channels[channel_id]
            dest_pe = branch.config.graph.vertices[channel.destination]

            assert cut_point.channel_states[channel_id].count() <= 1, "ill-formed initial dataflow config"
            if cut_point.channel_states[channel_id].ready():
                cut_point.channel_states[channel_id].pop()

            # If the destination is a carry, we need to simplify the decider condition to a constant
            if dest_pe.operator == "CF_CFG_OP_CARRY" and channel.destination_port == 0:
                # TODO: check if the channel value is always true under path condition
                # print(config.path_conditions)
                decider_term = branch.config.channel_states[channel_id].peek().term

                if smt.check_implication(branch.config.path_conditions, (smt.Equals(decider_term, smt.BVConst(1, dataflow.WORD_WIDTH)),)):
                    decider_value = smt.BVConst(1, dataflow.WORD_WIDTH)
                elif smt.check_implication(branch.config.path_conditions, (smt.Equals(decider_term, smt.BVConst(0, dataflow.WORD_WIDTH)),)):
                    decider_value = smt.BVConst(0, dataflow.WORD_WIDTH)
                else:
                    assert False, f"cannot determine a constant decider value {decider_term} from path conditions {branch.config.path_conditions}"

                cut_point.channel_states[channel_id].push(dataflow.PermissionedValue(decider_value, dummy_permission))

            else:
                # Generalize the value in the channel and assign an LLVM correspondence
                producer_llvm_var = self.find_non_steer_inv_producer_llvm_var(channel_id)

                if producer_llvm_var is not None:
                    sanitized_var = SimulationChecker.sanitize_llvm_name(producer_llvm_var)
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_var_{sanitized_var}_%d")
                    cut_point.channel_states[channel_id].push(dataflow.PermissionedValue(fresh_var, dummy_permission))

                    assert llvm_cut_point.has_variable(producer_llvm_var), \
                           f"corresponding llvm var {producer_llvm_var} of channel {channel_id} is not defined at the llvm cut point {branch.llvm_branch.to_cut_point}"
                    if producer_llvm_var not in llvm_var_correspondence:
                        llvm_var_correspondence[producer_llvm_var] = []
                    llvm_var_correspondence[producer_llvm_var].append(fresh_var)
                else:
                    if channel.source is None:
                        # An unused constant value
                        cut_point.channel_states[channel_id].push(dataflow.PermissionedValue(
                            branch.config.channel_states[channel_id].peek().term,
                            dummy_permission,
                        ))
                    else:
                        assert False, f"cannot find an llvm variable corresponding to a value in channel {channel.id}"

        # Construct a Correspondence object
        assert branch.llvm_branch.to_cut_point is not None
        correspondence = self.get_init_correspondence(cut_point, llvm_cut_point)
        correspondence.var_correspondence = tuple(
            (dataflow_var, llvm_cut_point.get_variable(llvm_var_name))
            for llvm_var_name, dataflow_vars in llvm_var_correspondence.items()
            for dataflow_var in dataflow_vars
        )

        return cut_point, correspondence

    def get_init_correspondence(self, dataflow_config: dataflow.Configuration, llvm_config: llvm.Configuration) -> Correspondence:
        return Correspondence(
            tuple(
                (dataflow_config.free_vars[param_name], llvm_config.get_variable("%" + param_name))
                for param_name, _ in self.dataflow_params.items()
            ),
            (dataflow_config.memory, llvm_config.memory),
            (),
        )
