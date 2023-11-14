
from __future__ import annotations

from typing import Set, Tuple, Iterable, List, Dict, Mapping, Optional, Union
from dataclasses import dataclass

from .graph import DataflowGraph, FunctionArgument

import semantics.smt as smt


class Term:
    def get_free_variables(self) -> Set[Variable]:
        raise NotImplementedError()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        raise NotImplementedError()

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Term:
        raise NotImplementedError()


@dataclass
class Empty(Term):
    def __str__(self):
        return "0"

    def get_free_variables(self) -> Set[Variable]:
        return set()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return self

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Term:
        return self


@dataclass
class Read(Term):
    heap_object: str

    def __str__(self):
        return f"read {self.heap_object}"

    def get_free_variables(self) -> Set[Variable]:
        return set()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return self

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Term:
        if self.heap_object in substitution:
            if substitution[self.heap_object] is None:
                return Empty()
            else:
                return Read(substitution[self.heap_object])
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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Term:
        if self.heap_object in substitution:
            if substitution[self.heap_object] is None:
                return Empty()
            else:
                return Write(substitution[self.heap_object])
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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Term:
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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Term:
        return DisjointUnion.of(*(term.substitute_heap_object(substitution) for term in self.terms))


class Formula:
    def get_free_variables(self) -> Set[Variable]:
        raise NotImplementedError()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Formula:
        raise NotImplementedError()

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Formula:
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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Formula:
        return Equality(self.left.substitute_heap_object(substitution), self.right.substitute_heap_object(substitution))

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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Formula:
        return Inclusion(self.left.substitute_heap_object(substitution), self.right.substitute_heap_object(substitution))

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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Formula:
        return Disjoint(tuple(term.substitute_heap_object(substitution) for term in self.terms))

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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Formula:
        return Conjunction(tuple(formula.substitute_heap_object(substitution) for formula in self.formulas))

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

    def substitute_heap_object(self, substitution: Mapping[str, Optional[str]]) -> Formula:
        return Disjunction(tuple(formula.substitute_heap_object(substitution) for formula in self.formulas))

    def is_atomic(self) -> bool:
        return False


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
                    subterms.append(Read(obj))

                if write_obj:
                    subterms.append(Write(obj))

            return DisjointUnion.of(*subterms)

        @staticmethod
        def get_zero(heap_objects: Tuple[str, ...]) -> RWPermissionPCM.Element:
            return RWPermissionPCM.Element(heap_objects, {
                obj: (smt.FALSE(), smt.FALSE())
                for obj in heap_objects
            })

        @staticmethod
        def get_read_atom(heap_objects: Tuple[str, ...], target_obj: str) -> RWPermissionPCM.Element:
            assert target_obj in heap_objects, f"invalid heap object {target_obj} (only given {heap_objects})"
            return RWPermissionPCM.Element(heap_objects, {
                obj: (smt.TRUE(), smt.FALSE()) if obj == target_obj else (smt.FALSE(), smt.FALSE())
                for obj in heap_objects
            })

        @staticmethod
        def get_write_atom(heap_objects: Tuple[str, ...], target_obj: str) -> RWPermissionPCM.Element:
            assert target_obj in heap_objects, f"invalid heap object {target_obj} (only given {heap_objects})"
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

    def interpret_term(self, assignment: Mapping[Variable, RWPermissionPCM.Element], term: Term) -> Tuple[RWPermissionPCM.Element, smt.SMTTerm]:
        """
        Interprete a term in the PCM, returning (permission element, condition to be well-defined)
        """

        if isinstance(term, Empty):
            return RWPermissionPCM.Element.get_zero(self.heap_objects), smt.TRUE()

        if isinstance(term, Read):
            return RWPermissionPCM.Element.get_read_atom(self.heap_objects, term.heap_object), smt.TRUE()

        if isinstance(term, Write):
            return RWPermissionPCM.Element.get_write_atom(self.heap_objects, term.heap_object), smt.TRUE()

        if isinstance(term, Variable):
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

    def interpret_formula(self, assignment: Mapping[Variable, RWPermissionPCM.Element], formula: Formula) -> smt.SMTTerm:
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
            return smt.And(interp, defined)

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
            return smt.And(interp, defined)

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
            return smt.And(interp, defined)

        if isinstance(formula, Conjunction):
            subformula_interps = tuple(self.interpret_formula(assignment, subformula) for subformula in formula.formulas)
            return smt.And(*subformula_interps)

        if isinstance(formula, Disjunction):
            subformula_interps = tuple(self.interpret_formula(assignment, subformula) for subformula in formula.formulas)
            return smt.Or(*subformula_interps)

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
            smt_var, defined = RWPermissionPCM.Element.get_free_variable(heap_objects)
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
