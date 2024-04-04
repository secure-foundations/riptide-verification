
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
class HasRead(Formula):
    """
    This is a special case of inclusion specifically to say that a read permission is in a permission term.
    """

    heap_object: str
    term: Term

    def __str__(self):
        return f"read {self.heap_object} ⊑ {self.term}"

    def get_free_variables(self) -> Set[Variable]:
        return self.term.get_free_variables()

    def substitute(self, substitution: Mapping[Variable, Term]) -> Term:
        return HasRead(self.heap_object, self.term.substitute(substitution))

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


class PermissionAlgebraElement:
    ...


class PermissionAlgebra:
    def get_value_from_smt_model(self, element: PermissionAlgebraElement, model: smt.SMTModel) -> Term:
        raise NotImplementedError()

    def get_fresh_variable(self) -> Tuple[PermissionAlgebraElement, smt.SMTTerm]:
        raise NotImplementedError()

    def interpret_term(self, assignment: Mapping[Variable, PermissionAlgebraElement], term: Term) -> Tuple[PermissionAlgebraElement, smt.SMTTerm]:
        raise NotImplementedError()

    def interpret_formula(self, assignment: Mapping[Variable, PermissionAlgebraElement], formula: Formula) -> smt.SMTTerm:
        raise NotImplementedError()


@dataclass
class FiniteFractionalPA(PermissionAlgebra):
    """
    Finite fractional permission algebra:
    Elements { read 0, read 1, ..., read n, write }
    """

    heap_objects: Tuple[str, ...]
    read_count: int # how many reads can a write be split into

    @dataclass
    class Element(PermissionAlgebraElement):
        smt_terms: Mapping[str, Tuple[smt.SMTTerm, ...]]

    def get_value_from_smt_model(self, element: FiniteFractionalPA.Element, model: smt.SMTModel) -> Term:
        # get a concrete interpretation from an SMT model
        # assuming self.smt_terms only contains variables

        subterms: List[Term] = []

        for obj in self.heap_objects:
            tentative_subterms: List[Term] = []
            is_write = True

            for read_index in range(self.read_count):
                read_obj = model[element.smt_terms[obj][read_index]].constant_value()

                if read_obj:
                    tentative_subterms.append(Read(obj, read_index))
                else:
                    is_write = False

            if is_write:
                subterms.append(Write(obj))
            else:
                subterms.extend(tentative_subterms)

        return DisjointUnion.of(*subterms)

    def get_zero(self) -> FiniteFractionalPA.Element:
        return FiniteFractionalPA.Element({
            obj: tuple(smt.FALSE() for _ in range(self.read_count))
            for obj in self.heap_objects
        })

    def get_read_atom(self, target_obj: str, read_index: int) -> FiniteFractionalPA.Element:
        assert target_obj in self.heap_objects, f"invalid heap object {target_obj} (only given {self.heap_objects})"
        return FiniteFractionalPA.Element({
            obj:
                tuple(
                    (smt.TRUE() if obj == target_obj and idx == read_index else smt.FALSE())
                    for idx in range(self.read_count)
                )
            for obj in self.heap_objects
        })

    def get_write_atom(self, target_obj: str) -> FiniteFractionalPA.Element:
        assert target_obj in self.heap_objects, f"invalid heap object {target_obj} (only given {self.heap_objects})"
        return FiniteFractionalPA.Element({
            obj: tuple(smt.TRUE() for _ in range(self.read_count)) if obj == target_obj else tuple(smt.FALSE() for _ in range(self.read_count))
            for obj in self.heap_objects
        })

    def get_fresh_variable(self) -> Tuple[FiniteFractionalPA.Element, smt.SMTTerm]:
        smt_vars = {
            obj: tuple(smt.FreshSymbol(smt.BOOL) for _ in range(self.read_count))
            for obj in self.heap_objects
        }

        defined = smt.TRUE()

        return FiniteFractionalPA.Element(smt_vars), defined

    def interpret_term(self, assignment: Mapping[Variable, FiniteFractionalPA.Element], term: Term) -> Tuple[FiniteFractionalPA.Element, smt.SMTTerm]:
        """
        Interprete a term in the PCM, returning (permission element, condition to be well-defined)
        """

        if isinstance(term, Empty):
            return self.get_zero(), smt.TRUE()

        if isinstance(term, Read):
            return self.get_read_atom(term.heap_object, term.index), smt.TRUE()

        if isinstance(term, Write):
            return self.get_write_atom(term.heap_object), smt.TRUE()

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
                    for index in range(self.read_count)
                ),
            )

            interp: Dict[str, Tuple[smt.SMTTerm, smt.SMTTerm]] = {
                obj: tuple(
                    smt.Or(subterm_interp.smt_terms[obj][index] for subterm_interp, _ in subterm_interps)
                    for index in range(self.read_count)
                )
                for obj in self.heap_objects
            }

            return FiniteFractionalPA.Element(interp), defined

        assert False, f"unsupported term {term}"

    def interpret_formula(self, assignment: Mapping[Variable, FiniteFractionalPA.Element], formula: Formula) -> smt.SMTTerm:
        """
        Interpret a formula in the PCM, returning (truth, condition to be well-defined)
        """

        if isinstance(formula, Equality):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.And(
                    smt.Iff(left_interp.smt_terms[obj][i], right_interp.smt_terms[obj][i])
                    for i in range(self.read_count)
                )
                for obj in self.heap_objects
            )
            defined = smt.And(left_defined, right_defined)
            return smt.And(interp, defined)

        if isinstance(formula, HasRead):
            term_interp, term_defined = self.interpret_term(assignment, formula.term)
            interp = smt.Or(
                term_interp.smt_terms[formula.heap_object][index]
                for index in range(self.read_count)
            )
            return smt.And(interp, term_defined)

        if isinstance(formula, Inclusion):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.Implies(
                    left_interp.smt_terms[obj][index],
                    right_interp.smt_terms[obj][index],
                )
                for obj in self.heap_objects
                for index in range(self.read_count)
            )
            defined = smt.And(left_defined, right_defined)
            return smt.And(interp, defined)

        if isinstance(formula, Disjoint):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in formula.terms)
            interp = smt.And(
                smt.AtMostOne(subterm_interp.smt_terms[obj][index] for subterm_interp, _ in subterm_interps)
                for obj in self.heap_objects
                for index in range(self.read_count)
            )
            defined = smt.And(cond for _, cond in subterm_interps)
            return smt.And(interp, defined)

        if isinstance(formula, Conjunction):
            return smt.And(self.interpret_formula(assignment, subformula) for subformula in formula.formulas)

        if isinstance(formula, Disjunction):
            return smt.Or(self.interpret_formula(assignment, subformula) for subformula in formula.formulas)

        assert False, f"unsupported formula {formula}"


@dataclass
class FiniteFractionalPAInteger(PermissionAlgebra):
    """
    Same as FiniteFractionalPA but encoded in integers
    """

    heap_objects: Tuple[str, ...]
    read_count: int # how many reads can a write be split into

    @dataclass
    class Element(PermissionAlgebraElement):
        smt_terms: Mapping[str, smt.SMTTerm]

    def get_value_from_smt_model(self, element: FiniteFractionalPAInteger.Element, model: smt.SMTModel) -> Term:
        # get a concrete interpretation from an SMT model
        # assuming self.smt_terms only contains variables

        subterms: List[Term] = []

        for obj in self.heap_objects:
            value = model[element.smt_terms[obj]].constant_value()
            assert value >= 0 and value <= self.read_count

            if value == self.read_count:
                subterms.append(Write(obj))
            else:
                for i in range(value):
                    subterms.append(Read(obj, i))

        return DisjointUnion.of(*subterms)

    def get_zero(self) -> FiniteFractionalPAInteger.Element:
        return FiniteFractionalPAInteger.Element({
            obj: smt.Int(0)
            for obj in self.heap_objects
        })

    def get_read_atom(self, target_obj: str, read_index: int) -> FiniteFractionalPAInteger.Element:
        assert target_obj in self.heap_objects, f"invalid heap object {target_obj} (only given {self.heap_objects})"
        return FiniteFractionalPAInteger.Element({
            obj: smt.Int(1) if obj == target_obj else smt.Int(0)
            for obj in self.heap_objects
        })

    def get_write_atom(self, target_obj: str) -> FiniteFractionalPAInteger.Element:
        assert target_obj in self.heap_objects, f"invalid heap object {target_obj} (only given {self.heap_objects})"
        return FiniteFractionalPAInteger.Element({
            obj: smt.Int(self.read_count) if obj == target_obj else smt.Int(0)
            for obj in self.heap_objects
        })

    def get_fresh_variable(self) -> Tuple[FiniteFractionalPAInteger.Element, smt.SMTTerm]:
        smt_vars = {
            obj: smt.FreshSymbol(smt.INT)
            for obj in self.heap_objects
        }

        defined = smt.And(
            smt.And(
                smt.GE(smt_vars[obj], smt.Int(0)),
                smt.LE(smt_vars[obj], smt.Int(self.read_count)),
            )
            for obj in self.heap_objects
        )

        return FiniteFractionalPAInteger.Element(smt_vars), defined

    def interpret_term(self, assignment: Mapping[Variable, FiniteFractionalPAInteger.Element], term: Term) -> Tuple[FiniteFractionalPAInteger.Element, smt.SMTTerm]:
        """
        Interprete a term in the PCM, returning (permission element, condition to be well-defined)
        """

        if isinstance(term, Empty):
            return self.get_zero(), smt.TRUE()

        if isinstance(term, Read):
            return self.get_read_atom(term.heap_object, term.index), smt.TRUE()

        if isinstance(term, Write):
            return self.get_write_atom(term.heap_object), smt.TRUE()

        if isinstance(term, Variable):
            assert term in assignment, f"unable ot find variable {term} in the given assignment"
            return assignment[term], smt.TRUE()

        if isinstance(term, DisjointUnion):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in term.terms)
            defined = smt.And(
                smt.And(cond for _, cond in subterm_interps),
                smt.And(
                    smt.LE(
                        smt.Plus(subterm_interp.smt_terms[obj] for subterm_interp, _ in subterm_interps),
                        smt.Int(self.read_count),
                    )
                    for obj in self.heap_objects
                ),
            )

            interp: Dict[str, Tuple[smt.SMTTerm, smt.SMTTerm]] = {
                obj: smt.Plus(subterm_interp.smt_terms[obj] for subterm_interp, _ in subterm_interps)
                for obj in self.heap_objects
            }

            return FiniteFractionalPAInteger.Element(interp), defined

        assert False, f"unsupported term {term}"

    def interpret_formula(self, assignment: Mapping[Variable, FiniteFractionalPAInteger.Element], formula: Formula) -> smt.SMTTerm:
        """
        Interpret a formula in the PCM, returning (truth, condition to be well-defined)
        """

        if isinstance(formula, Equality):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.Equals(left_interp.smt_terms[obj], right_interp.smt_terms[obj])
                for obj in self.heap_objects
            )
            defined = smt.And(left_defined, right_defined)
            return smt.And(interp, defined)

        if isinstance(formula, HasRead):
            term_interp, term_defined = self.interpret_term(assignment, formula.term)
            interp = smt.GT(term_interp.smt_terms[formula.heap_object], smt.Int(0))
            return smt.And(interp, term_defined)

        if isinstance(formula, Inclusion):
            left_interp, left_defined = self.interpret_term(assignment, formula.left)
            right_interp, right_defined = self.interpret_term(assignment, formula.right)
            interp = smt.And(
                smt.LE(
                    left_interp.smt_terms[obj],
                    right_interp.smt_terms[obj],
                )
                for obj in self.heap_objects
            )
            defined = smt.And(left_defined, right_defined)
            return smt.And(interp, defined)

        if isinstance(formula, Disjoint):
            subterm_interps = tuple(self.interpret_term(assignment, subterm) for subterm in formula.terms)
            interp = smt.And(
                smt.LE(
                    smt.Plus(subterm_interp.smt_terms[obj] for subterm_interp, _ in subterm_interps),
                    smt.Int(self.read_count),
                )
                for obj in self.heap_objects
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
        perm_algebra: PermissionAlgebra,
        constraints: Iterable[Formula],
        unsat_core: bool = False, # generate an unsat core if unsat
    ) -> SolverResult:
        free_vars: Set[Variable] = set()

        for constraint in constraints:
            free_vars.update(constraint.get_free_variables())

        # perm_algebra = FiniteFractionalPA(heap_objects)

        assignment: Dict[Variable, PermissionAlgebraElement] = {}
        assignment_defined = smt.TRUE()

        for var in free_vars:
            smt_var, defined = perm_algebra.get_fresh_variable()
            assignment[var] = smt_var
            assignment_defined = smt.And(assignment_defined, defined)

        formula_to_constraint: Dict[smt.SMTTerm, Union[str, Formula]] = {}

        with smt.UnsatCoreSolver(name="z3") if unsat_core else smt.Solver(name="z3") as solver:
            solver.add_assertion(assignment_defined)

            if unsat_core:
                formula_to_constraint[assignment_defined] = "(definedness of free variables)"

            for constraint in constraints:
                valid = perm_algebra.interpret_formula(assignment, constraint)
                solver.add_assertion(valid)

                if unsat_core:
                    formula_to_constraint[valid] = constraint

            if solver.solve():
                model = solver.get_model()
                solution = {
                    var: perm_algebra.get_value_from_smt_model(assignment[var], model)
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
