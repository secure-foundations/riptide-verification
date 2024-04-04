from __future__ import annotations

from typing import Tuple, Iterable, Optional, List, OrderedDict, Dict, Union
from dataclasses import dataclass
from collections import OrderedDict

import time
import logging

import semantics.smt as smt
from semantics.matching import *

import semantics.dataflow as dataflow
import semantics.llvm as llvm


logger = logging.getLogger(__name__)


PERMISSION_PREFIX_CUT_POINT = "cut-point-"
PERMISSION_PREFIX_CHANNEL = "channel-"
PERMISSION_PREFIX_EXPANSION = "expansion-"


@dataclass
class LoopHeaderHint:
    block_name: str
    incoming_block: str
    back_edge_block: str


class CutPointPlacement:
    """
    Strategy to generate cut points for LLVM
    """

    def __init__(self, llvm_function: llvm.Function, loop_header_hints: Iterable[LoopHeaderHint]):
        self.llvm_function = llvm_function
        self.loop_header_hints = tuple(loop_header_hints)

    def gen_llvm_cut_points(self) -> Tuple[llvm.Configuration, ...]:
        raise NotImplementedError()


class BackEdgeOnly(CutPointPlacement):
    """
    One cut point per loop back edge
    """

    def gen_llvm_cut_points(self) -> Tuple[llvm.Configuration, ...]:
        cut_points = []

        for header_info in self.loop_header_hints:
            # if header_info.block_name == "for.cond2":
            #     continue

            llvm_cut_point = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)
            llvm_cut_point.current_block = header_info.block_name
            llvm_cut_point.previous_block = header_info.back_edge_block
            cut_points.append(llvm_cut_point)

        # llvm_cut_point = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)
        # llvm_cut_point.current_block = "if.then"
        # llvm_cut_point.previous_block = "for.body6"
        # cut_points.append(llvm_cut_point)

        # llvm_cut_point = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)
        # llvm_cut_point.current_block = "for.inc"
        # llvm_cut_point.previous_block = "for.body6"
        # cut_points.append(llvm_cut_point)

        return tuple(cut_points)


class IncomingAndBackEdge(CutPointPlacement):
    """
    Two cut points for each loop header: incoming and back edge
    """

    def gen_llvm_cut_points(self) -> Tuple[llvm.Configuration, ...]:
        cut_points = []

        for header_info in self.loop_header_hints:
            llvm_cut_point = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)
            llvm_cut_point.current_block = header_info.block_name
            llvm_cut_point.previous_block = header_info.incoming_block
            cut_points.append(llvm_cut_point)

            llvm_cut_point = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function)
            llvm_cut_point.current_block = header_info.block_name
            llvm_cut_point.previous_block = header_info.back_edge_block
            cut_points.append(llvm_cut_point)

        return tuple(cut_points)


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
    permission_equalities: Optional[Tuple[dataflow.permission.Equality, ...]] # None for final configuration
    source_expansion_index: Optional[int] = None


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
        cut_point_placement: CutPointPlacement,
        permission_fractional_reads: int = 4,
        permission_unsat_core: bool = False,
        cut_point_expansion: bool = True,
    ):
        self.solver = smt.Solver(name="z3", random_seed=0)
        self.permission_fractional_reads = permission_fractional_reads
        self.permission_unsat_core = permission_unsat_core
        self.cut_point_expansion = cut_point_expansion

        self.dataflow_graph = dataflow_graph
        self.llvm_function = llvm_function

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

        # Find all merge gates
        self.merge_pe_ids = tuple(
            pe.id
            for pe in dataflow_graph.vertices
            if pe.operator == "CF_CFG_OP_MERGE"
        )

        # Set up initial configs
        assert len(self.llvm_function.parameters) == len(self.dataflow_graph.function_arguments)

        self.dataflow_params: OrderedDict[str, smt.SMTTerm] = OrderedDict(
            (function_arg.variable_name, smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_param_{function_arg.variable_name}_%d"))
            for function_arg in self.dataflow_graph.function_arguments
        )

        dataflow_init_config = dataflow.Configuration.get_initial_configuration(
            self.dataflow_graph,
            self.dataflow_params,
            self.solver,
            permission_prefix=f"{PERMISSION_PREFIX_CUT_POINT}0-",
        )
        llvm_init_config = llvm.Configuration.get_initial_configuration(self.llvm_function.module, self.llvm_function, self.solver)

        self.llvm_cut_points: List[llvm.Configuration] = [ llvm_init_config, *cut_point_placement.gen_llvm_cut_points() ]
        self.num_cut_points = len(self.llvm_cut_points)

        for i, llvm_cut_point in enumerate(self.llvm_cut_points):
            logger.debug(f"llvm cut point {i}: block {llvm_cut_point.current_block}, prev {llvm_cut_point.previous_block}")
            llvm_cut_point.solver = self.solver

        self.dataflow_cut_points: List[Optional[dataflow.Configuration]] = [ dataflow_init_config ] + [ None ] * (self.num_cut_points - 1)
        self.correspondence: List[Optional[Correspondence]] = \
            [ self.get_init_correspondence(dataflow_init_config, llvm_init_config) ] + [ None ] * (self.num_cut_points - 1)

        self.num_cut_points = len(self.dataflow_cut_points)
        self.dataflow_cut_points_executed: List[bool] = [ False ] * self.num_cut_points

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

        self.base_pointer_mapping: Dict[str, str] = {}
        self.heap_objects: List[str] = []

        # Find the set of heap obejcts
        has_aliasing = False
        for parameter in self.llvm_function.parameters.values():
            if not isinstance(parameter.get_type(), llvm.PointerType):
                continue

            if parameter.is_noalias():
                self.heap_objects.append(parameter.name)
                self.base_pointer_mapping[parameter.name[1:]] = parameter.name
            else:
                # Could alias with each other, so we gather them all into other_mem
                if not has_aliasing:
                    has_aliasing = True
                    self.heap_objects.append("other_mem")
                self.base_pointer_mapping[parameter.name[1:]] = "other_mem"

        self.permission_var_counter = 0

    def get_fresh_permission_var(self, prefix: str) -> dataflow.permission.Variable:
        return dataflow.permission.GlobalPermissionVarCounter.get_fresh_permission_var(prefix)

    def get_pe_id_from_llvm_position(self, position: Tuple[str, int]) -> Optional[int]:
        return self.llvm_position_to_pe_id.get(position)

    def match_llvm_branches(self) -> None:
        """
        Run llvm cut points and match the branches against all cut points (or final configs)
        This will fill up self.matched_llvm_branches and self.final_llvm_branches
        """
        for i, llvm_cut_point in enumerate(self.llvm_cut_points):
            logger.debug(f"[llvm] running cut point {i}")

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
                                assert match_result.check_condition(self.solver), f"invalid match at cut point {j}"
                                logger.debug(f"[llvm] found a matching config to cut point {j}")
                                self.matched_llvm_branches[j][i].append(LLVMBranch(result.config, match_result, new_trace, i, j))
                                break
                        else:
                            # continue execution
                            queue.append((result.config, new_trace))

                    elif isinstance(result, llvm.FunctionReturn):
                        logger.debug("[llvm] found a final config")
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
            if smt.check_implication(branch_conditions, left.path_conditions, self.solver):
                left_branches.append(branch)
            elif smt.check_implication(branch_conditions, right.path_conditions, self.solver):
                right_branches.append(branch)
            else:
                blame_left = smt.find_implication_blame(branch_conditions, left.path_conditions, self.solver)
                blame_right = smt.find_implication_blame(branch_conditions, right.path_conditions, self.solver)
                logger.debug(f"correspondence: {correspondence}")
                logger.debug(f"llvm branch path conditions: {branch.config.path_conditions}")
                logger.debug(f"left blame: {blame_left}")
                logger.debug(f"right blame: {blame_right}")
                assert False, f"failed to categorize a llvm branch into neither dataflow branches"

        return left_branches, right_branches

    def find_nested_merges(self, pe_id: int) -> Tuple[int, ...]:
        """
        Return a tuple of nested merge gates (not including the given pe)
        """
        pe = self.dataflow_graph.vertices[pe_id]
        if pe.operator != "CF_CFG_OP_MERGE":
            return ()
        else:
            if pe.inputs[1].source is None:
                input_1_merges = ()
            else:
                input_1_merges = self.find_nested_merges(pe.inputs[1].source)

            if pe.inputs[2].source is None:
                input_2_merges = ()
            else:
                input_2_merges = self.find_nested_merges(pe.inputs[2].source)

            return input_1_merges + input_2_merges + (pe_id,)

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
                results = config.step_until_branch(self.steer_inv_pe_ids, base_pointer_mapping=self.base_pointer_mapping)
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
                    base_pointer_mapping=self.base_pointer_mapping,
                )
                if len(results) == 1:
                    changed = True
                    config = results[0].config
                elif len(results) > 1:
                    return branch(results)

                # Step on all merge gates in the initial state
                results = config.step_until_branch(
                    (
                        pe_id
                        for pe_id in self.merge_pe_ids
                        if config.operator_states[pe_id].current_transition == dataflow.MergeOperator.start
                    ),
                    exhaust=False,
                    base_pointer_mapping=self.base_pointer_mapping,
                )
                if len(results) == 1:
                    changed = True
                    config = results[0].config
                elif len(results) > 1:
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
                return DataflowBranch(config, None, llvm_branches[0], None),

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

            scheduled_pe_ids = (pe_id,)
            pe = self.dataflow_graph.vertices[pe_id]
            if pe.operator == "CF_CFG_OP_MERGE":
                # Check for nested merges
                scheduled_pe_ids = self.find_nested_merges(pe.id)

            # logger.debug(f"executing {scheduled_pe_ids}")
            results = config.step_until_branch(scheduled_pe_ids, base_pointer_mapping=self.base_pointer_mapping)
            if len(results) == 1:
                config = results[0].config
            elif len(results) > 1:
                return branch(results)
            else:
                logger.debug(config)
                assert False, f"PE {pe_id} {scheduled_pe_ids} corresponding to llvm instruction {position} not ready when scheduled to fire"

            dataflow_branches = run_misc_operators()
            if dataflow_branches:
                return dataflow_branches

    def find_non_steer_inv_producer(self, channel_id: int) -> Optional[Union[dataflow.ProcessingElement, dataflow.Constant]]:
        """
        Find the non-steer and non-inv producer of a channel
        """

        channel = self.dataflow_graph.channels[channel_id]
        if channel.source is None:
            assert channel.constant is not None
            return channel.constant

        source_pe = self.dataflow_graph.vertices[channel.source]

        if source_pe.operator == "CF_CFG_OP_STEER" or source_pe.operator == "CF_CFG_OP_INVARIANT":
            return self.find_non_steer_inv_producer(source_pe.inputs[1].id)
        else:
            return source_pe

    def get_defined_llvm_var_of_pe(self, pe_id: int) -> str:
        pe = self.dataflow_graph.vertices[pe_id]
        if pe.llvm_position is None:
            return None

        # Find the corresponding llvm var
        block_name, instr_index = pe.llvm_position
        llvm_instr = self.llvm_function.blocks[block_name].instructions[instr_index]
        defined_var = llvm_instr.get_defined_variable()
        return defined_var

    def refresh_exec_permission_vars(self, constraints: Iterable[dataflow.permission.Formula]) -> Tuple[dataflow.permission.Formula, ...]:
        """
        Replace all {PERMISSION_PREFIX_EXEC}* permission vars with fresh ones
        """
        constraints = tuple(constraints)

        free_vars = set()
        substitution: OrderedDict[dataflow.permission.Variable, dataflow.permission.Variable] = OrderedDict()

        for constraint in constraints:
            free_vars.update(constraint.get_free_variables())

        for free_var in sorted(tuple(free_vars), key=lambda v: v.name):
            if not free_var.name.startswith(PERMISSION_PREFIX_CUT_POINT):
                assert free_var.name.startswith(dataflow.PERMISSION_PREFIX_EXEC), f"unexpected free var {free_var.name}"
                prefix = "-".join(free_var.name.split("-")[:-1]) + "-"
                substitution[free_var] = self.get_fresh_permission_var(prefix)

        return tuple(constraint.substitute(substitution) for constraint in constraints)

    def replace_source_expansion_permission_vars(self, branch: DataflowBranch, target_expansion_index: Optional[int]) -> Tuple[dataflow.permission.Formula, ...]:
        """
        Prepend original cut point variables with the source_expansion_index prefix
        and also refresh all exec- variables

        Prepend target cut point match variables with target_expansion_index prefix
        (if specified)
        """

        # Make a conjunction to simplify substitution
        permission_constraint = dataflow.permission.Conjunction(tuple(branch.config.permission_constraints))

        source_index = branch.llvm_branch.from_cut_point
        target_index = branch.llvm_branch.to_cut_point

        rhs_substitution = {}

        # Rename source cut point vars and refresh exec vars
        for free_var in sorted(tuple(permission_constraint.get_free_variables()), key=lambda v: v.name):
            if free_var.name.startswith(f"{PERMISSION_PREFIX_CUT_POINT}{source_index}"):
                # need consistent renaming here
                rhs_substitution[free_var] = dataflow.permission.Variable(f"{PERMISSION_PREFIX_EXPANSION}{branch.source_expansion_index}-" + free_var.name)
            else:
                # refresh other variables
                assert free_var.name.startswith(dataflow.PERMISSION_PREFIX_EXEC), \
                       f"unexpected free var {free_var.name} from cut point {source_index} to {target_index or '⊥'}"
                prefix = "-".join(free_var.name.split("-")[:-1]) + "-"
                rhs_substitution[free_var] = self.get_fresh_permission_var(prefix)

        substituted_permission_constraints = permission_constraint.substitute(rhs_substitution).formulas

        # Add equality constraints
        if target_expansion_index is not None:
            assert target_index is not None
            # Create the constraints to match with one of the target_expansion_index
            # print(substituted_equality_constraint)
            lhs_substitution = {}
            for free_var in self.dataflow_cut_points[target_index].get_free_permission_vars():
                assert free_var.name.startswith(f"{PERMISSION_PREFIX_CUT_POINT}{target_index}")
                lhs_substitution[free_var] = dataflow.permission.Variable(f"{PERMISSION_PREFIX_EXPANSION}{target_expansion_index}-" + free_var.name)

            substituted_permission_constraints += tuple(
                dataflow.permission.Equality(equality.left.substitute(lhs_substitution), equality.right.substitute(rhs_substitution))
                for equality in branch.permission_equalities
            )

        else:
            assert branch.permission_equalities is None

        return substituted_permission_constraints

    def analyze_permission_unsat_core(
        self,
        unsat_core: Tuple[dataflow.permission.Formula, ...],
        constraint_to_branch_indices: Dict[int, Tuple[int, Optional[int], Optional[int], Optional[int]]],
    ):
        """
        Analyze the unsat core and output debug information
        """

        branch_indices_to_constraints: Dict[Tuple[int, Optional[int], Optional[int], Optional[int]], List[dataflow.permission.Formula]] = {}

        for constraint in unsat_core:
            branch_indices = constraint_to_branch_indices[id(constraint)]
            if branch_indices not in branch_indices_to_constraints:
                branch_indices_to_constraints[branch_indices] = []
            branch_indices_to_constraints[branch_indices].append(constraint)

        logger.debug("dumping unsat core")

        for branch_indices, constraints in branch_indices_to_constraints.items():

            if branch_indices[2] is None:
                print(f"### constrains from {branch_indices[0]}@{branch_indices[1]} to ⊥")
            else:
                print(f"### constrains from {branch_indices[0]}@{branch_indices[1]} to {branch_indices[2]}@{branch_indices[3]}")

            disjoint_constraints = []
            equality_constraints = []
            linearity_constraints = []
            rw_constraints = []
            other_constraints = []

            for constraint in constraints:
                if isinstance(constraint, dataflow.permission.Disjoint):
                    disjoint_constraints.append(constraint)

                elif isinstance(constraint, dataflow.permission.Equality):
                    equality_constraints.append(constraint)

                elif isinstance(constraint, dataflow.permission.Inclusion):
                    if isinstance(constraint.left, dataflow.permission.Read) or \
                       isinstance(constraint.left, dataflow.permission.Write) or \
                       (isinstance(constraint.left, dataflow.permission.DisjointUnion) and \
                        len(constraint.left.terms) == 1 and \
                        (isinstance(constraint.left.terms[0], dataflow.permission.Read) or \
                         isinstance(constraint.left.terms[0], dataflow.permission.Write))):
                        rw_constraints.append(constraint)
                    else:
                        linearity_constraints.append(constraint)

                else:
                    other_constraints.append(constraint)

            print("\n\n".join([
                "\n".join(["# disjoints"] + list(map(str, disjoint_constraints))),
                "\n".join(["# equalities"] + list(map(str, equality_constraints))),
                "\n".join(["# linearity"] + list(map(str, linearity_constraints))),
                "\n".join(["# rw"] + list(map(str, rw_constraints))),
                "\n".join(["# others"] + list(map(str, other_constraints))),
            ]))

        print("# unsat core size:", len(unsat_core))

    def check_confluence(self):
        """
        Check the satifiability of the conjunction of all memory permission constraints
        """

        # cut point index |-> expansion size
        expansion_size: List[int] = []
        expansion_index_to_branch: List[List[DataflowBranch]] = [ [] for j in range(self.num_cut_points) ]

        if self.cut_point_expansion:
            # Mark each branch from each cut point with a unique source expansion index
            for j in range(self.num_cut_points):
                num_non_final_branches = sum(len(self.matched_dataflow_branches[i][j]) for i in range(self.num_cut_points))
                num_final_branches = len(self.final_dataflow_branches[j])

                assert num_non_final_branches + num_final_branches >= 1
                expansion_size.append(num_non_final_branches + num_final_branches)

                for i in range(self.num_cut_points):
                    for dataflow_branch in self.matched_dataflow_branches[i][j]:
                        dataflow_branch.source_expansion_index = len(expansion_index_to_branch[j])
                        expansion_index_to_branch[j].append(dataflow_branch)

                for dataflow_branch in self.final_dataflow_branches[j]:
                    dataflow_branch.source_expansion_index = len(expansion_index_to_branch[j])
                    expansion_index_to_branch[j].append(dataflow_branch)

            logger.debug(f"using cut point expansion sizes {expansion_size}")
        else:
            logger.debug("cut point expansion disabled")

        constraints: List[dataflow.permission.Formula] = []

        # for debug purpose, indices are (source cut point index, source expansion index, target cut point index, target expansion index)
        constraint_to_branch_indices: Dict[int, Tuple[int, Optional[int], Optional[int], Optional[int]]] = {}

        for j in range(self.num_cut_points):
            for i in range(self.num_cut_points):
                for dataflow_branch in self.matched_dataflow_branches[i][j]:
                    if self.cut_point_expansion:
                        # Expand one cut point to expansion_size[j] many cut points
                        # Allow permission tokens to be assigned more freely

                        for target_expansion_index in range(expansion_size[i]):
                            # Check if this branch into the expanded cut point is feasible
                            # dataflow_branch.config.path_conditions
                            # dataflow_branch.match_result
                            # expansion_index_to_branch[i][target_expansion_index].config.path_conditions
                            if not smt.check_sat([
                                *dataflow_branch.config.path_conditions,
                                *(
                                    target_path_condition.substitute(dataflow_branch.match_result.substitution)
                                    for target_path_condition in expansion_index_to_branch[i][target_expansion_index].config.path_conditions
                                ),
                            ]):
                                logger.debug(f"pruned a branch from {j}@{dataflow_branch.source_expansion_index} to {i}@{target_expansion_index}")
                                continue

                            logger.debug(f"adding confluence constraints for a branch from {j}@{dataflow_branch.source_expansion_index} to {i}@{target_expansion_index}")

                            permission_constraints = self.replace_source_expansion_permission_vars(dataflow_branch, target_expansion_index)
                            constraints.extend(permission_constraints)

                            for constraint in permission_constraints:
                                constraint_to_branch_indices[id(constraint)] = j, dataflow_branch.source_expansion_index, i, target_expansion_index

                    else:
                        logger.debug(f"adding confluence constraints for a branch from {j} to {i}")
                        permission_constraints = tuple(dataflow_branch.config.permission_constraints) + tuple(dataflow_branch.permission_equalities)
                        permission_constraints = self.refresh_exec_permission_vars(permission_constraints)
                        constraints.extend(permission_constraints)

                        for constraint in permission_constraints:
                            constraint_to_branch_indices[id(constraint)] = j, None, i, None

            for dataflow_branch in self.final_dataflow_branches[j]:
                if self.cut_point_expansion:
                    logger.debug(f"adding confluence constraints for a branch from {j}@{dataflow_branch.source_expansion_index} to ⊥")
                    permission_constraints = self.replace_source_expansion_permission_vars(dataflow_branch, None)
                    constraints.extend(permission_constraints)

                    for constraint in permission_constraints:
                        constraint_to_branch_indices[id(constraint)] = j, dataflow_branch.source_expansion_index, None, None

                else:
                    logger.debug(f"adding confluence constraints for a branch from {j} to ⊥")
                    permission_constraints = self.refresh_exec_permission_vars(dataflow_branch.config.permission_constraints)
                    constraints.extend(permission_constraints)

                    for constraint in permission_constraints:
                        constraint_to_branch_indices[id(constraint)] = j, None, None, None

        # for i, constraint in enumerate(constraints):
        #     constraints[i] = constraint.substitute_heap_object(heap_object_substitution)
        #     constraint_to_branch_indices[id(constraints[i])] = constraint_to_branch_indices[id(constraint)]

        # for constraint in constraints:
        #     print(constraint)

        logger.debug(f"base pointer mapping: {self.base_pointer_mapping}")
        logger.debug(f"heap objects: {self.heap_objects}")
        logger.debug(f"checking sat of {len(constraints)} memory permission constraints")

        perm_algebra = dataflow.permission.FiniteFractionalPA(tuple(self.heap_objects), self.permission_fractional_reads)
        result = dataflow.permission.PermissionSolver.solve_constraints(
            perm_algebra,
            constraints,
            unsat_core=self.permission_unsat_core,
        )
        if isinstance(result, dataflow.permission.ResultUnsat):
            logger.warning("confluence check result: unsat - may not be confluent")

            if self.permission_unsat_core:
                unsat_core = result.unsat_core
                assert unsat_core is not None
                self.analyze_permission_unsat_core(unsat_core, constraint_to_branch_indices)

        else:
            assert isinstance(result, dataflow.permission.ResultSat)
            logger.debug("confluence check result: sat - confluent")

            # for var, term in result.solution.items():
            #     print(f"{var} = {term}")

    def check_branch_bisimulation_obligation(self, dataflow_branch: DataflowBranch):
        llvm_branch = dataflow_branch.llvm_branch
        logger.debug(f"checking bisimulation obligations for a branch from cut point {llvm_branch.from_cut_point} to {llvm_branch.to_cut_point or '⊥'}")

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

        if not smt.check_implication(source_correspondence_smt, obligations, self.solver):
            blame = smt.find_implication_blame(source_correspondence_smt, obligations, self.solver)
            logger.debug(f"knows: {source_correspondence}")
            logger.debug("blame:\n" + "\n".join(map(lambda t: "  " + t.serialize(), blame)))
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

        logger.debug("bisim check succeeds")

    def check_dataflow_matches(self):
        """
        Check if the branches in self.matched_dataflow_branches actually match
        their corresponding cut points. If so set branch.match_result

        Returns a list of permission equalities that are required for the matchings to work
        """

        for i in range(self.num_cut_points):
            target_dataflow_cut_point = self.dataflow_cut_points[i]

            for j in range(self.num_cut_points):
                for dataflow_branch in self.matched_dataflow_branches[i][j]:
                    logger.debug(f"[dataflow] checking a matched branch from cut point {j} to {i}")
                    match_result, permission_equalities = target_dataflow_cut_point.match(dataflow_branch.config)
                    # print(f"llvm trace from {dataflow_branch.llvm_branch.from_cut_point} to {dataflow_branch.llvm_branch.to_cut_point}: {dataflow_branch.llvm_branch.trace}")
                    assert isinstance(match_result, MatchingSuccess), \
                           f"failed to match an expected dataflow branch from cut point {j} to {i}: {match_result.reason}"
                    assert match_result.check_condition(self.solver), "unexpected matching failure"
                    dataflow_branch.match_result = match_result
                    dataflow_branch.permission_equalities = permission_equalities

        # Check that final configs actually terminates
        for i in range(self.num_cut_points):
            for dataflow_branch in self.final_dataflow_branches[i]:
                logger.debug(f"[dataflow] checking that a final branch from {i} is not fireable")

                for pe in self.dataflow_graph.vertices:
                    if dataflow_branch.config.is_fireable(pe.id):
                        logger.warning(f"[dataflow] non-terminating final state: PE {pe.id} is still fireable")

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

        logger.debug(f"[dataflow] mirroring llvm cut point {cut_point_index}")

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
                    logger.debug(f"[dataflow] inferring dataflow cut point {target_cut_point} using a dataflow trace from cut point {cut_point_index}")
                    target_dataflow_cut_point, target_correspondence = self.generalize_dataflow_branch_to_cut_point(target_cut_point, dataflow_branch)
                    logger.debug(f"[dataflow] inferred dataflow cut point {target_cut_point}\n{target_dataflow_cut_point}")
                    logger.debug(f"[dataflow] inferred correspondence at cut point {target_cut_point}\n{target_correspondence}")

                    self.dataflow_cut_points[target_cut_point] = target_dataflow_cut_point
                    self.correspondence[target_cut_point] = target_correspondence

                else:
                    # TODO: check if the generalized config matches anyway?
                    ...
            else:
                self.final_dataflow_branches[cut_point_index].append(dataflow_branch)

    def generalize_dataflow_branch_to_cut_point(self, target_cut_point_index: int, branch: DataflowBranch) -> Tuple[dataflow.Configuration, Correspondence]:
        """
        Generalize a dataflow config to a cut point.

        Returns the generalized template and a correspondence
        """
        permission_prefix = f"{PERMISSION_PREFIX_CUT_POINT}{target_cut_point_index}-"

        # not copying self.dataflow_cut_points[0] since we want fresh permission variables
        cut_point = dataflow.Configuration.get_initial_configuration(self.dataflow_graph, self.dataflow_params, self.solver, permission_prefix)

        # Mapping from llvm var name |-> generalized dataflow variables corresponding to it
        llvm_var_correspondence: OrderedDict[str, List[smt.SMTTerm]] = OrderedDict()

        llvm_cut_point = self.llvm_cut_points[branch.llvm_branch.to_cut_point]

        # Mirror the operator states
        for pe_id, operator in enumerate(branch.config.operator_states):
            cut_point.operator_states[pe_id].transition_to(operator.current_transition)

            if isinstance(operator, dataflow.InvariantOperator) and \
               operator.current_transition == dataflow.InvariantOperator.loop:
                input_channel_id = self.dataflow_graph.vertices[pe_id].inputs[1].id
                producer = self.find_non_steer_inv_producer(input_channel_id)

                if isinstance(producer, dataflow.ProcessingElement):
                    producer_llvm_var = self.get_defined_llvm_var_of_pe(producer.id)
                    assert producer_llvm_var is not None, f"producer {producer.id} found corresponding to an invariant value in PE {pe_id} has no LLVM annotation"

                    sanitized_var = SimulationChecker.sanitize_llvm_name(producer_llvm_var)
                    fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_var_{sanitized_var}_%d")
                    cut_point.operator_states[pe_id].value = fresh_var

                    assert llvm_cut_point.has_variable(producer_llvm_var), \
                           f"corresponding llvm var {producer_llvm_var} of an invariant value at PE {pe_id} is not defined at the llvm cut point {branch.llvm_branch.to_cut_point}"
                    if producer_llvm_var not in llvm_var_correspondence:
                        llvm_var_correspondence[producer_llvm_var] = []
                    llvm_var_correspondence[producer_llvm_var].append(fresh_var)

                else:
                    assert isinstance(producer, dataflow.Constant)
                    if isinstance(producer, dataflow.FunctionArgument):
                        inv_value = cut_point.free_vars[producer.variable_name]
                    else:
                        assert isinstance(producer, dataflow.ConstantValue)
                        inv_value = smt.BVConst(producer.value, dataflow.WORD_WIDTH)

                    cut_point.operator_states[pe_id].value = inv_value

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

            channel = self.dataflow_graph.channels[channel_id]

            # Generalize the value in the channel and assign an LLVM correspondence
            producer = self.find_non_steer_inv_producer(channel_id)

            if isinstance(producer, dataflow.ProcessingElement):
                producer_llvm_var = self.get_defined_llvm_var_of_pe(producer.id)
                assert producer_llvm_var is not None, f"producer {producer.id} found corresponding to a non-constant value in channel {channel_id} has no LLVM annotation"
                assert not cut_point.channel_states[channel_id].ready()

                sanitized_var = SimulationChecker.sanitize_llvm_name(producer_llvm_var)
                fresh_var = smt.FreshSymbol(smt.BVType(dataflow.WORD_WIDTH), f"dataflow_var_{sanitized_var}_%d")
                perm = self.get_fresh_permission_var(f"{permission_prefix}{PERMISSION_PREFIX_CHANNEL}{channel_id}-")
                cut_point.channel_states[channel_id].push(dataflow.PermissionedValue(fresh_var, perm))

                assert llvm_cut_point.has_variable(producer_llvm_var), \
                        f"corresponding llvm var {producer_llvm_var} of channel " \
                        f"{channel_id} is not defined at the llvm cut point {branch.llvm_branch.to_cut_point}"
                if producer_llvm_var not in llvm_var_correspondence:
                    llvm_var_correspondence[producer_llvm_var] = []
                llvm_var_correspondence[producer_llvm_var].append(fresh_var)
            else:
                assert isinstance(producer, dataflow.Constant)
                if isinstance(producer, dataflow.FunctionArgument):
                    channel_value = cut_point.free_vars[producer.variable_name]
                else:
                    assert isinstance(producer, dataflow.ConstantValue)
                    channel_value = smt.BVConst(producer.value, dataflow.WORD_WIDTH)

                if cut_point.channel_states[channel_id].ready():
                    cut_point.channel_states[channel_id].pop()

                perm = self.get_fresh_permission_var(f"{permission_prefix}{PERMISSION_PREFIX_CHANNEL}{channel_id}-")
                cut_point.channel_states[channel_id].push(dataflow.PermissionedValue(channel_value, perm))

        # Reset the permission constraints to include the permissions we created
        cut_point.permission_constraints = dataflow.Configuration.get_initial_permission_constraints(cut_point)

        # Construct a Correspondence object
        assert branch.llvm_branch.to_cut_point is not None
        correspondence = self.get_init_correspondence(cut_point, llvm_cut_point)

        # If there's a bit width mismatch, shrink the dataflow var to the bit width of the llvm var
        temp_var_correspondence: List[Tuple[smt.SMTTerm, smt.SMTTerm]] = []
        for llvm_var_name, dataflow_vars in llvm_var_correspondence.items():
            bit_width = self.llvm_function.definitions[llvm_var_name].get_type().get_bit_width()
            assert bit_width <= dataflow.WORD_WIDTH

            for dataflow_var in dataflow_vars:
                llvm_var = llvm_cut_point.get_variable(llvm_var_name)
                if bit_width < dataflow.WORD_WIDTH:
                    llvm_var = smt.BVZExt(llvm_var, dataflow.WORD_WIDTH - bit_width)

                temp_var_correspondence.append(((dataflow_var, llvm_var)))

        correspondence.var_correspondence = tuple(temp_var_correspondence)

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

    def run_all_checks(self):
        start_time = time.process_time()
        self.match_llvm_branches()
        self.generate_dataflow_cut_points()
        self.check_dataflow_matches()
        self.check_bisimulation()
        elapsed = time.process_time() - start_time
        logger.debug(f"bisim check took {round(elapsed, 2)}s")

        start_time = time.process_time()
        self.check_confluence()
        elapsed = time.process_time() - start_time
        logger.debug(f"confluence check took {round(elapsed, 2)}s")
