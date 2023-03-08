
from __future__ import annotations

from typing import Set, Tuple, Iterable, List, Dict, Mapping
from dataclasses import dataclass

from graph import DataflowGraph, FunctionArgument, Channel, ProcessingElement

import smt


class Term:
    def get_free_variables(self) -> Set[PermissionVariable]:
        raise NotImplementedError()


@dataclass
class EmptyPermission(Term):
    def __str__(self):
        return "0"

    def get_free_variables(self) -> Set[PermissionVariable]:
        return set()


@dataclass
class ReadPermission(Term):
    heap_object: str

    def __str__(self):
        return f"read {self.heap_object}"

    def get_free_variables(self) -> Set[PermissionVariable]:
        return set()


@dataclass
class WritePermission(Term):
    heap_object: str

    def __str__(self):
        return f"write {self.heap_object}"

    def get_free_variables(self) -> Set[PermissionVariable]:
        return set()


@dataclass(frozen=True)
class PermissionVariable(Term):
    """
    Permission for a channel
    """
    channel_id: int

    def __str__(self):
        return f"p{self.channel_id}"

    def get_free_variables(self) -> Set[PermissionVariable]:
        return {self}


@dataclass
class DisjointUnion(Term):
    terms: Tuple[Term, ...]

    def __str__(self):
        return " + ".join(map(str, self.terms))
    
    def get_free_variables(self) -> Set[PermissionVariable]:
        return set().union(*(term.get_free_variables() for term in self.terms))

    @staticmethod
    def of(*terms: Term) -> DisjointUnion:
        terms = tuple(terms)

        if len(terms) == 0:
            return EmptyPermission()

        return DisjointUnion(terms)


class Formula:
    def get_free_variables(self) -> Set[PermissionVariable]:
        raise NotImplementedError()


@dataclass
class Equality(Formula):
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} = {self.right}"

    def get_free_variables(self) -> Set[PermissionVariable]:
        return self.left.get_free_variables().union(self.right.get_free_variables())


@dataclass
class Inclusion(Formula):
    """
    Permission left is included in permission right
    """
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} ⊑ {self.right}"

    def get_free_variables(self) -> Set[PermissionVariable]:
        return self.left.get_free_variables().union(self.right.get_free_variables())


@dataclass
class Disjoint(Formula):
    """
    The given permission terms are disjoint
    """
    terms: Tuple[Term, ...]

    def get_free_variables(self) -> Set[PermissionVariable]:
        return set().union(*(term.get_free_variables() for term in self.terms))

    def __str__(self):
        return f"disjoint({', '.join(map(str, self.terms))})"


class RWPermissionPCM:
    """
    A concrete description of the permission PCM in SMT
    
    ## Permission PCM
    Given a set H of disjoint heap object names, the permission partial commutative monoid P(H)
    consists of the following elements:
      - Additive unit 0
      - read A for any A in H
      - write A for any A in H
      - p1 + p2 for any terms p1 and p2
    modulo the following axioms:
      - read A + read A = read A
      - associativity of +
      - commutativity of +
      - 0 + p = p + 0 = p

    Furthermore, we also remove any element containing
      - read A + write A
      - write A + write A

    ## Representing the PCM in SMT/SAT
    For finite H, we encode each permission p in P(H) as a list of Boolean values.
    For instance, suppose H = { A, B },
    each element of P(H) is encoded as 2 * |H| Boolean values each corresponding to { read A, write A, read B, write B }:
    read A |-> { 1, 0, 0, 0 }
    read B |-> { 0, 0, 1, 0 }
    write A |-> { 0, 1, 0, 0 }
    read A + write B |-> { 1, 0, 0, 1 }
    read A + write A |-> { 1, 1, 0, 0 } does not exist
    """

    class Element:
        def __init__(self, heap_objects: Tuple[str, ...], smt_terms: Mapping[str, Tuple[smt.SMTTerm, smt.SMTTerm]]):
            self.heap_objects = heap_objects
            self.smt_terms = dict(smt_terms)

        def get_value_from_smt_model(self, model: smt.SMTModel) -> Term:
            # get a concrete interpretation from an SMT model
            # assuming self.smt_terms only contains variables

            subterms: List[Term] = []

            for obj in self.heap_objects:
                read_obj = model[self.smt_terms[obj][0]].constant_value()
                write_obj = model[self.smt_terms[obj][1]].constant_value()
                assert not (read_obj and write_obj), "a valid permission should not have both read and write to a heap object"

                if read_obj:
                    subterms.append(ReadPermission(obj))

                if write_obj:
                    subterms.append(WritePermission(obj))
            
            return DisjointUnion.of(*subterms)

        @staticmethod
        def get_zero(heap_objects: Tuple[str, ...]) -> RWPermissionPCM.Element:
            return RWPermissionPCM.Element(heap_objects, {
                obj: (smt.FALSE(), smt.FALSE())
                for obj in heap_objects
            })

        @staticmethod
        def get_read_atom(heap_objects: Tuple[str, ...], target_obj: str) -> RWPermissionPCM.Element:
            return RWPermissionPCM.Element(heap_objects, {
                obj: (smt.TRUE(), smt.FALSE()) if obj == target_obj else (smt.FALSE(), smt.FALSE())
                for obj in heap_objects
            })

        @staticmethod
        def get_write_atom(heap_objects: Tuple[str, ...], target_obj: str) -> RWPermissionPCM.Element:
            return RWPermissionPCM.Element(heap_objects, {
                obj: (smt.FALSE(), smt.TRUE()) if obj == target_obj else (smt.FALSE(), smt.FALSE())
                for obj in heap_objects
            })

        @staticmethod
        def get_free_variable(heap_objects: Tuple[str, ...]) -> Tuple[RWPermissionPCM.Element, smt.SMTTerm]:
            smt_vars = {
                obj: (smt.FreshSymbol(smt.BOOL), smt.FreshSymbol(smt.BOOL))
                for obj in heap_objects
            }

            defined = smt.And(
                smt.Not(smt.And(smt_vars[obj][0], smt_vars[obj][1]))
                for obj in heap_objects
            )

            return RWPermissionPCM.Element(heap_objects, smt_vars), defined

    def __init__(self, heap_objects: Tuple[str, ...]):
        self.heap_objects = heap_objects

    def interpret_term(self, assignment: Mapping[PermissionVariable, RWPermissionPCM.Element], term: Term) -> Tuple[RWPermissionPCM.Element, smt.SMTTerm]:
        """
        Interprete a term in the PCM, returning (permission element, condition to be well-defined)
        """

        if isinstance(term, EmptyPermission):
            return RWPermissionPCM.Element.get_zero(self.heap_objects), smt.TRUE()

        if isinstance(term, ReadPermission):
            return RWPermissionPCM.Element.get_read_atom(self.heap_objects, term.heap_object), smt.TRUE()

        if isinstance(term, WritePermission):
            return RWPermissionPCM.Element.get_write_atom(self.heap_objects, term.heap_object), smt.TRUE()

        if isinstance(term, PermissionVariable):
            assert term in assignment, f"unable ot find variable {term} in the given assignment"
            return assignment[term], smt.TRUE()

        if isinstance(term, DisjointUnion):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in term.terms)
            defined = smt.And(cond for _, cond in subterm_interps)

            interp: Dict[str, Tuple[smt.SMTTerm, smt.SMTTerm]] = {}

            for obj in self.heap_objects:
                defined = smt.And(
                    defined,
                    smt.Or(
                        # no write
                        smt.Not(smt.Or(subterm_interp.smt_terms[obj][1] for subterm_interp, _ in subterm_interps)),

                        # exactly one write and no read
                        smt.And(
                            smt.ExactlyOne(subterm_interp.smt_terms[obj][1] for subterm_interp, _ in subterm_interps),
                            smt.Not(smt.Or(subterm_interp.smt_terms[obj][0] for subterm_interp, _ in subterm_interps))
                        ),
                    ),
                )

                interp[obj] = (
                    smt.Or(subterm_interp.smt_terms[obj][0] for subterm_interp, _ in subterm_interps),
                    smt.Or(subterm_interp.smt_terms[obj][1] for subterm_interp, _ in subterm_interps),
                )

            return RWPermissionPCM.Element(self.heap_objects, interp), defined

        assert False, f"unsupported term {term}"

    def interpret_formula(self, assignment: Mapping[PermissionVariable, RWPermissionPCM.Element], formula: Formula) -> Tuple[smt.SMTTerm, smt.SMTTerm]:
        """
        Interpret a formula in the PCM, returning (truth, condition to be well-defined)
        """

        if isinstance(formula, Equality):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.And(
                    smt.Iff(left_interp.smt_terms[obj][0], right_interp.smt_terms[obj][0]),
                    smt.Iff(left_interp.smt_terms[obj][1], right_interp.smt_terms[obj][1]),
                )
                for obj in self.heap_objects
            )
            defined = smt.And(left_defined, right_defined)
            return interp, defined

        if isinstance(formula, Inclusion):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.Or(
                    right_interp.smt_terms[obj][1],
                    smt.And(right_interp.smt_terms[obj][0], smt.Not(left_interp.smt_terms[obj][1])),
                    smt.And(smt.Not(left_interp.smt_terms[obj][0]), smt.Not(left_interp.smt_terms[obj][1])),
                )
                for obj in self.heap_objects
            )
            defined = smt.And(left_defined, right_defined)
            return interp, defined

        if isinstance(formula, Disjoint):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in formula.terms)
            interp = smt.And(
                smt.Or(
                    # no write
                    smt.Not(smt.Or(subterm_interp.smt_terms[obj][1] for subterm_interp, _ in subterm_interps)),

                    # exactly one write and no read
                    smt.And(
                        smt.ExactlyOne(subterm_interp.smt_terms[obj][1] for subterm_interp, _ in subterm_interps),
                        smt.Not(smt.Or(subterm_interp.smt_terms[obj][0] for subterm_interp, _ in subterm_interps))
                    ),
                )
                for obj in self.heap_objects
            )
            defined = smt.And(cond for _, cond in subterm_interps)
            return interp, defined

        assert False, f"unsupported formula {formula}"


class MemoryPermissionSolver:
    @staticmethod
    def get_sum_of_channel_permissions(channels: Iterable[Channel]) -> Term:
        return DisjointUnion.of(*map(lambda c: PermissionVariable(c.id), channels))

    @staticmethod
    def get_heap_object_of_memory_operator(pe: ProcessingElement) -> str:
        # TODO: here I'm making an assumption that array accesses
        # would always have the first input as the array variable
        # assert len(pe.inputs) == 3
        array_input = pe.inputs[0]
        assert isinstance(array_input.constant, FunctionArgument)
        return array_input.constant.variable_name

    @staticmethod
    def generate_constraints(graph: DataflowGraph) -> Tuple[Tuple[str, ...], Tuple[Formula, ...]]:
        heap_objects: Set[str] = set() # names of heap objects

        # we assume that these heap objects do not alias
        # i.e. when they are declared in the source file
        # they should have the "restrict" modifier

        for pe in graph.vertices:
            if pe.operator == "MEM_CFG_OP_LOAD" or pe.operator == "MEM_CFG_OP_STORE":
                heap_objects.add(MemoryPermissionSolver.get_heap_object_of_memory_operator(pe))

        # Constraints:
        # 1. (Affinity) for each operator, the sum of input permissions contains the sum of output permissions,
        #    and the input permissions are disjoint.
        # 2. (Read permission) for a load operator on heap object A, (read A) must be in one of the input permissions.
        # 3. (Write permission) for a store operator on heap object B, (write A) must be in one of the input permissions.
        # 4. Channels with hold=true (infinite number of constant values) must have empty permission.
        #    NOTE: this condition might be too strong
        # 5. Channels with a single initial constant value should have mutually disjoint memory permissions.

        print("found heap objects", heap_objects)

        constraints: List[Formula] = []

        for pe in graph.vertices:
            input_sum = MemoryPermissionSolver.get_sum_of_channel_permissions(pe.inputs)
            output_sum = MemoryPermissionSolver.get_sum_of_channel_permissions(sum(pe.outputs.values(), ()))

            if pe.operator == "CF_CFG_OP_CARRY":
                assert len(pe.inputs) == 3
                decider, input_a, input_b = pe.inputs
                constraints.append(Inclusion(output_sum, MemoryPermissionSolver.get_sum_of_channel_permissions((decider, input_a))))
                constraints.append(Inclusion(output_sum, MemoryPermissionSolver.get_sum_of_channel_permissions((decider, input_b))))
            else:
                # default constraint NOTE: may be too weak!
                constraints.append(Inclusion(output_sum, input_sum))

            if pe.operator == "MEM_CFG_OP_LOAD":
                heap_object = MemoryPermissionSolver.get_heap_object_of_memory_operator(pe)
                constraints.append(Inclusion(WritePermission(heap_object), input_sum))
                
            elif pe.operator == "MEM_CFG_OP_STORE":
                heap_object = MemoryPermissionSolver.get_heap_object_of_memory_operator(pe)
                constraints.append(Inclusion(WritePermission(heap_object), input_sum))

        for channel in graph.channels:
            if channel.hold:
                constraints.append(Equality(PermissionVariable(channel.id), EmptyPermission()))

        constant_channel_vars: List[PermissionVariable] = []

        for channel in graph.channels:
            if channel.constant is not None and not channel.hold:
                constant_channel_vars.append(PermissionVariable(channel.id))

        constraints.append(Disjoint(constant_channel_vars))

        return tuple(heap_objects), constraints

    @staticmethod
    def solve_constraints(heap_objects: Tuple[str, ...], constraints: Iterable[Formula]) -> Optional[Dict[PermissionVariable, Term]]:
        free_vars: Set[PermissionVariable] = set()

        for constraint in constraints:
            free_vars.update(constraint.get_free_variables())

        assignment: Dict[PermissionVariable, RWPermissionPCM.Element] = {}
        assignment_defined = smt.TRUE()

        for var in free_vars:
            smt_var, defined = RWPermissionPCM.Element.get_free_variable(heap_objects)
            assignment[var] = smt_var
            assignment_defined = smt.And(assignment_defined, defined)

        pcm = RWPermissionPCM(heap_objects)
        solution: Dict[PermissionVariable, Term] = {}

        with smt.Solver(name="z3") as solver:
            solver.add_assertion(assignment_defined)

            for constraint in constraints:
                valid, defined = pcm.interpret_formula(assignment, constraint)
                solver.add_assertion(valid)
                solver.add_assertion(defined)

            if solver.solve():
                model = solver.get_model()

                print("found a solution:")
                for var in free_vars:
                    term = assignment[var].get_value_from_smt_model(model)
                    solution[var] = term
                    print(f"  {var} = {term}")

                return solution

            else:
                print("failed to find a solution")
                return None
