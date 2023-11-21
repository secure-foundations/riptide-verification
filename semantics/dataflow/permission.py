
from __future__ import annotations

from typing import Set, Tuple, Iterable, List, Dict, Mapping, Optional, Union
from dataclasses import dataclass

from .graph import DataflowGraph, FunctionArgument

import semantics.smt as smt


# How many reads can a write be split into
READ_COUNT = 4


class Term:
    def get_free_variables(self) -> Set[Variable]:
        raise NotImplementedError()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        raise NotImplementedError()


@dataclass
class Empty(Term):
    def __str__(self):
        return "0"

    def get_free_variables(self) -> Set[Variable]:
        return set()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return self


@dataclass
class Read(Term):
    heap_object: str
    index: int # [0, READ_COUNT)

    def __str__(self):
        return f"read ({self.index}) {self.heap_object}"

    def get_free_variables(self) -> Set[Variable]:
        return set()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return self


@dataclass
class Write(Term):
    heap_object: str

    def __str__(self):
        return f"write {self.heap_object}"

    def get_free_variables(self) -> Set[Variable]:
        return set()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return self


@dataclass(frozen=True)
class Variable(Term):
    """
    Permission for a channel
    """
    name: str

    def __str__(self):
        return f"p({self.name})"

    def get_free_variables(self) -> Set[Variable]:
        return {self}

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        if self in substitution:
            return substitution[self]
        return self


@dataclass
class DisjointUnion(Term):
    terms: Tuple[Term, ...]

    def __str__(self):
        return " + ".join(map(str, self.terms))

    @staticmethod
    def of(*terms: Term) -> DisjointUnion:
        terms = tuple(terms)

        if len(terms) == 0:
            return Empty()

        return DisjointUnion(terms)

    def get_free_variables(self) -> Set[Variable]:
        return set().union(*(term.get_free_variables() for term in self.terms))

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return DisjointUnion.of(*(term.substitute(substitution) for term in self.terms))


class Formula:
    def get_free_variables(self) -> Set[Variable]:
        raise NotImplementedError()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Formula:
        raise NotImplementedError()

    def is_atomic(self) -> bool:
        raise NotImplementedError()


@dataclass
class Equality(Formula):
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} = {self.right}"

    def get_free_variables(self) -> Set[Variable]:
        return self.left.get_free_variables().union(self.right.get_free_variables())

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return Equality(self.left.substitute(substitution), self.right.substitute(substitution))

    def is_atomic(self) -> bool:
        return True


@dataclass
class Inclusion(Formula):
    """
    Permission left is included in permission right
    """
    left: Term
    right: Term

    def __str__(self):
        return f"{self.left} ⊑ {self.right}"

    def get_free_variables(self) -> Set[Variable]:
        return self.left.get_free_variables().union(self.right.get_free_variables())

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return Inclusion(self.left.substitute(substitution), self.right.substitute(substitution))

    def is_atomic(self) -> bool:
        return True


@dataclass
class Disjoint(Formula):
    """
    The given permission terms are disjoint
    """
    terms: Tuple[Term, ...]

    def __str__(self):
        return f"disjoint({', '.join(map(str, self.terms))})"

    def get_free_variables(self) -> Set[Variable]:
        return set().union(*(term.get_free_variables() for term in self.terms))

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return Disjoint(tuple(term.substitute(substitution) for term in self.terms))

    def is_atomic(self) -> bool:
        return True


@dataclass
class Conjunction(Formula):
    formulas: Tuple[Formula, ...]

    def __str__(self):
        return " /\\ ".join(map(lambda f: str(f) if f.is_atomic() else f"({f})", self.formulas))

    def get_free_variables(self) -> Set[Variable]:
        return set().union(*(formula.get_free_variables() for formula in self.formulas))

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return Conjunction(tuple(formula.substitute(substitution) for formula in self.formulas))

    def is_atomic(self) -> bool:
        return False


@dataclass
class Disjunction(Formula):
    formulas: Tuple[Formula, ...]

    def __str__(self):
        return " \\/ ".join(map(lambda f: str(f) if f.is_atomic() else f"({f})", self.formulas))

    def get_free_variables(self) -> Set[Variable]:
        return set().union(*(formula.get_free_variables() for formula in self.formulas))

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return Disjunction(tuple(formula.substitute(substitution) for formula in self.formulas))

    def is_atomic(self) -> bool:
        return False


class PermissionAlgebra:
    ...


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
        def __init__(self, heap_objects: Tuple[str, ...], smt_terms: Mapping[str, Tuple[smt.SMTTerm, ...]]):
            self.heap_objects = heap_objects
            self.smt_terms = dict(smt_terms)

        def get_value_from_smt_model(self, model: smt.SMTModel) -> Term:
            # get a concrete interpretation from an SMT model
            # assuming self.smt_terms only contains variables

            subterms: List[Term] = []

            for obj in self.heap_objects:
                tentative_subterms: List[Term] = []
                is_write = True

                for read_index in range(READ_COUNT):
                    read_obj = model[self.smt_terms[obj][read_index]].constant_value()

                    if read_obj:
                        tentative_subterms.append(Read(obj, read_index))
                    else:
                        is_write = False

                if is_write:
                    subterms.append(Write(obj))
                else:
                    subterms.extend(tentative_subterms)

            return DisjointUnion.of(*subterms)

        @staticmethod
        def get_zero(heap_objects: Tuple[str, ...]) -> RWPermissionPCM.Element:
            return RWPermissionPCM.Element(heap_objects, {
                obj: tuple(smt.FALSE() for _ in range(READ_COUNT))
                for obj in heap_objects
            })

        @staticmethod
        def get_read_atom(heap_objects: Tuple[str, ...], target_obj: str, read_index: int) -> RWPermissionPCM.Element:
            assert target_obj in heap_objects, f"invalid heap object {target_obj} (only given {heap_objects})"
            return RWPermissionPCM.Element(heap_objects, {
                obj:
                    tuple(
                        (smt.TRUE() if obj == target_obj and idx == read_index else smt.FALSE())
                        for idx in range(READ_COUNT)
                    )
                for obj in heap_objects
            })

        @staticmethod
        def get_write_atom(heap_objects: Tuple[str, ...], target_obj: str) -> RWPermissionPCM.Element:
            assert target_obj in heap_objects, f"invalid heap object {target_obj} (only given {heap_objects})"
            return RWPermissionPCM.Element(heap_objects, {
                obj: tuple(smt.TRUE() for _ in range(READ_COUNT)) if obj == target_obj else tuple(smt.FALSE() for _ in range(READ_COUNT))
                for obj in heap_objects
            })

        @staticmethod
        def get_fresh_variable(heap_objects: Tuple[str, ...]) -> Tuple[RWPermissionPCM.Element, smt.SMTTerm]:
            smt_vars = {
                obj: tuple(smt.FreshSymbol(smt.BOOL) for _ in range(READ_COUNT))
                for obj in heap_objects
            }

            defined = smt.TRUE()

            return RWPermissionPCM.Element(heap_objects, smt_vars), defined

    def __init__(self, heap_objects: Tuple[str, ...]):
        self.heap_objects = heap_objects

    def interpret_term(self, assignment: Mapping[Variable, RWPermissionPCM.Element], term: Term) -> Tuple[RWPermissionPCM.Element, smt.SMTTerm]:
        """
        Interprete a term in the PCM, returning (permission element, condition to be well-defined)
        """

        if isinstance(term, Empty):
            return RWPermissionPCM.Element.get_zero(self.heap_objects), smt.TRUE()

        if isinstance(term, Read):
            return RWPermissionPCM.Element.get_read_atom(self.heap_objects, term.heap_object, term.index), smt.TRUE()

        if isinstance(term, Write):
            return RWPermissionPCM.Element.get_write_atom(self.heap_objects, term.heap_object), smt.TRUE()

        if isinstance(term, Variable):
            assert term in assignment, f"unable ot find variable {term} in the given assignment"
            return assignment[term], smt.TRUE()

        if isinstance(term, DisjointUnion):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in term.terms)
            defined = smt.And(
                smt.And(cond for _, cond in subterm_interps),
                smt.And(
                    smt.AtMostOne(subterm_interp.smt_terms[obj][index] for subterm_interp, _ in subterm_interps)
                    for obj in self.heap_objects
                    for index in range(READ_COUNT)
                ),
            )

            interp: Dict[str, Tuple[smt.SMTTerm, smt.SMTTerm]] = {
                obj: tuple(
                    smt.Or(subterm_interp.smt_terms[obj][index] for subterm_interp, _ in subterm_interps)
                    for index in range(READ_COUNT)
                )
                for obj in self.heap_objects
            }

            return RWPermissionPCM.Element(self.heap_objects, interp), defined

        assert False, f"unsupported term {term}"

    def interpret_formula(self, assignment: Mapping[Variable, RWPermissionPCM.Element], formula: Formula) -> smt.SMTTerm:
        """
        Interpret a formula in the PCM, returning (truth, condition to be well-defined)
        """

        if isinstance(formula, Equality):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.And(
                    smt.Iff(left_interp.smt_terms[obj][i], right_interp.smt_terms[obj][i])
                    for i in range(READ_COUNT)
                )
                for obj in self.heap_objects
            )
            defined = smt.And(left_defined, right_defined)
            return smt.And(interp, defined)

        if isinstance(formula, Inclusion):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.Implies(
                    left_interp.smt_terms[obj][index],
                    right_interp.smt_terms[obj][index],
                )
                for obj in self.heap_objects
                for index in range(READ_COUNT)
            )
            defined = smt.And(left_defined, right_defined)
            return smt.And(interp, defined)

        if isinstance(formula, Disjoint):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in formula.terms)
            interp = smt.And(
                smt.AtMostOne(subterm_interp.smt_terms[obj][index] for subterm_interp, _ in subterm_interps)
                for obj in self.heap_objects
                for index in range(READ_COUNT)
            )
            defined = smt.And(cond for _, cond in subterm_interps)
            return smt.And(interp, defined)

        if isinstance(formula, Conjunction):
            return smt.And(self.interpret_formula(assignment, subformula) for subformula in formula.formulas)

        if isinstance(formula, Disjunction):
            return smt.Or(self.interpret_formula(assignment, subformula) for subformula in formula.formulas)

        assert False, f"unsupported formula {formula}"


class SolverResult: ...


@dataclass
class ResultUnsat(SolverResult):
    unsat_core: Optional[Tuple[Formula, ...]]


@dataclass
class ResultSat(SolverResult):
    solution: Dict[Variable, Term]


class PermissionSolver:
    @staticmethod
    def find_function_argument_producers(graph: DataflowGraph, channel_id: int) -> Tuple[str, ...]:
        """
        Find the constant producer of the channel modulo +, gep, inv, steer, carry, merge, select
        """
        channel = graph.channels[channel_id]

        if channel.constant:
            if isinstance(channel.constant, FunctionArgument):
                return channel.constant.variable_name,
            else:
                return ()

        assert channel.source is not None
        source_pe = graph.vertices[channel.source]

        if source_pe.operator in { "CF_CFG_OP_STEER", "CF_CFG_OP_INVARIANT", "CF_CFG_OP_CARRY" }:
            return PermissionSolver.find_function_argument_producers(graph, source_pe.inputs[1].id)

        elif source_pe.operator in { "CF_CFG_OP_SELECT", "CF_CFG_OP_MERGE" }:
            return PermissionSolver.find_function_argument_producers(graph, source_pe.inputs[1].id) + \
                   PermissionSolver.find_function_argument_producers(graph, source_pe.inputs[2].id)

        elif source_pe.operator in { "ARITH_CFG_OP_ADD", "ARITH_CFG_OP_GEP" }:
            return PermissionSolver.find_function_argument_producers(graph, source_pe.inputs[0].id) + \
                   PermissionSolver.find_function_argument_producers(graph, source_pe.inputs[1].id)

        return ()

    @staticmethod
    def get_static_heap_objects(graph: DataflowGraph) -> Tuple[str, ...]:
        """
        Read all static heap objects from a graph
        """

        found = set()
        heap_objects = []

        for pe in graph.vertices:
            if pe.operator in ("MEM_CFG_OP_LOAD", "MEM_CFG_OP_STORE"):
                for name in PermissionSolver.find_function_argument_producers(graph, pe.inputs[0].id):
                    if name not in found:
                        found.add(name)
                        heap_objects.append(name)

        return tuple(heap_objects)

    @staticmethod
    def solve_constraints(
        heap_objects: Tuple[str, ...],
        constraints: Iterable[Formula],
        unsat_core: bool = False, # generate an unsat core if unsat
    ) -> SolverResult:
        free_vars: Set[Variable] = set()

        for constraint in constraints:
            free_vars.update(constraint.get_free_variables())

        assignment: Dict[Variable, RWPermissionPCM.Element] = {}
        assignment_defined = smt.TRUE()

        for var in free_vars:
            smt_var, defined = RWPermissionPCM.Element.get_fresh_variable(heap_objects)
            assignment[var] = smt_var
            assignment_defined = smt.And(assignment_defined, defined)

        pcm = RWPermissionPCM(heap_objects)

        formula_to_constraint: Dict[smt.SMTTerm, Union[str, Formula]] = {}

        with smt.UnsatCoreSolver(name="z3") if unsat_core else smt.Solver(name="z3") as solver:
            solver.add_assertion(assignment_defined)

            if unsat_core:
                formula_to_constraint[assignment_defined] = "(definedness of free variables)"

            for constraint in constraints:
                valid = pcm.interpret_formula(assignment, constraint)
                solver.add_assertion(valid)

                if unsat_core:
                    formula_to_constraint[valid] = constraint

            if solver.solve():
                model = solver.get_model()
                solution = {
                    var: assignment[var].get_value_from_smt_model(model)
                    for var in free_vars
                }

                return ResultSat(solution)

            else:
                if unsat_core:
                    unsat_core = tuple(
                        formula_to_constraint[unsat_core_formula]
                        for unsat_core_formula in solver.get_unsat_core()
                        if not isinstance(formula_to_constraint[unsat_core_formula], str)
                    )
                else:
                    unsat_core = None

                return ResultUnsat(unsat_core)


class GlobalPermissionVarCounter:
    counter: int = 0

    @staticmethod
    def get_fresh_permission_var(prefix: str = "p") -> Variable:
        var = Variable(f"{prefix}{GlobalPermissionVarCounter.counter}")
        GlobalPermissionVarCounter.counter += 1
        return var
