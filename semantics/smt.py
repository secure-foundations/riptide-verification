from typing import Any, Callable, Tuple, Generator, Iterable, Optional, List

from contextlib import contextmanager

from pysmt.shortcuts import ( # type: ignore
    FreshSymbol, Symbol,
    TRUE, FALSE, And, Or, Not, Xor, ExactlyOne, AtMostOne, Implies, Iff, ForAll, Exists, Ite, Equals,
    GT, GE, LT, LE,
    BV, Int, Bool,
    get_model, get_unsat_core, Solver, UnsatCoreSolver,
    Array, Select,
    get_free_variables,
    Plus, Store, Times,
    BVAdd, BVSub, BVMul, BVZExt, BVSExt, BVExtract, BVConcat, BVSGE, BVSGT, BVSLE, BVSLT, BVUGE, BVUGT, BVULE, BVULT, BVNeg,
    BVAnd, BVOr, BVXor, BVLShl, BVLShr, BVAShr,
)
from pysmt.typing import BOOL, INT, STRING, BVType, FunctionType, ArrayType, Type # type: ignore
from pysmt.smtlib.parser import SmtLibParser # type: ignore
from pysmt.oracles import QuantifierOracle # type: ignore

from pysmt.shortcuts import Function as Apply
from pysmt.shortcuts import get_env, reset_env

import io


SMTTerm = Any
SMTSolver = Any
SMTModel = Any
SMTSort = Any
SMTVariable = Any
SMTFunction = Callable[..., SMTTerm]
SMTScript = Any


_fresh_sort_counter = 0


def BVConst(value: int, width: int) -> SMTTerm:
    """
    Similar to smt.BV, but supports negative integers too
    """

    if value < 0:
        return BVNeg(BV(-value, width))
    else:
        return BV(value, width)

def reset() -> None:
    """
    (Re-)initialize global state
    """
    global _fresh_sort_counter
    _fresh_sort_counter = 0
    reset_env()


def FreshFunction(input_sorts: Tuple[SMTSort, ...], output_sort: SMTSort) -> SMTFunction:
    symbol = FreshSymbol(FunctionType(output_sort, input_sorts))
    return lambda *args: Apply(symbol, args)


def FreshSort() -> SMTSort:
    global _fresh_sort_counter
    name = f"FreshSort{_fresh_sort_counter}"
    _fresh_sort_counter += 1
    return Type(name)


def check_sat(terms: Iterable[SMTTerm], solver: Optional[Solver] = None) -> bool:
    """
    Check satisfiability of the conjunction of terms
    """
    if solver is None:
        with Solver(name="z3") as solver:
            return check_sat(terms, solver)
    else:
        with push_solver(solver):
            for term in terms:
                solver.add_assertion(term)
            return solver.solve()


def check_implication(a: Iterable[SMTTerm], b: Iterable[SMTTerm], solver: Optional[Solver] = None) -> bool:
    """
    Check the validity of the implication a => b
    """
    if solver is None:
        with Solver(name="z3") as solver:
            return check_implication(a, b, solver)
    else:
        with push_solver(solver):
            for term in a:
                solver.add_assertion(term)
            solver.add_assertion(Not(And(*b)))
            return not solver.solve()


def find_implication_blame(a: Iterable[SMTTerm], b: Iterable[SMTTerm], solver: Optional[Solver] = None) -> Tuple[SMTTerm, ...]:
    """
    find a list of terms in b that are not implied by a
    """
    if solver is None:
        with Solver(name="z3") as solver:
            return find_implication_blame(a, b, solver)
    else:
        with push_solver(solver):
            for term in a:
                solver.add_assertion(term)

            blame: List[SMTTerm] = []

            for term in b:
                with push_solver(solver):
                    solver.add_assertion(Not(term))
                    if solver.solve():
                        blame.append(term)

            return blame


@contextmanager
def push_solver(solver: Solver, clear_formula_manager: bool = False) -> Generator[None, None, None]:
    if clear_formula_manager:
        # NOTE: PySMT keeps track of all formulae every created
        # in a formula manager. This might create memory leaks
        # in certain use patterns.

        env = get_env()

        old_disable_memoization = env.disable_memoization
        env.disable_memoization = True

        old_formulae = dict(env.formula_manager.formulae)
        old_symbols = dict(env.formula_manager.symbols)
        old_int_constants = dict(env.formula_manager.int_constants)
        old_real_constants = dict(env.formula_manager.real_constants)
        old_string_constants = dict(env.formula_manager.string_constants)

    try:
        solver.push()
        yield
    finally:
        solver.pop()

        if clear_formula_manager:
            env.disable_memoization = old_disable_memoization
            env.formula_manager.formulae.clear()
            env.formula_manager.symbols.clear()
            env.formula_manager.int_constants.clear()

            env.formula_manager.formulae = old_formulae
            env.formula_manager.symbols = old_symbols
            env.formula_manager.int_constants = old_int_constants
            env.formula_manager.real_constants = old_real_constants
            env.formula_manager.string_constants = old_string_constants


def is_qfree(formula: SMTTerm) -> bool:
    return QuantifierOracle().is_qf(formula) # type: ignore


class SMTLIB:
    @staticmethod
    def parse_sort(src: str) -> SMTSort:
        variable = FreshSymbol().to_smtlib()
        return SmtLibParser().get_script(io.StringIO(f"(declare-const .{variable} {src}) (assert .{variable})")).get_last_formula().get_type()

    @staticmethod
    def parse_script(src: str) -> SMTScript:
        return SmtLibParser().get_script(io.StringIO(src))

    @staticmethod
    def parse_smt_function_from_template(src: str) -> SMTFunction:
        """
        Given a template such as (+ #1 #2)
        return a function that maps (x, y) |-> (+ x y)
        NOTE: indices start with 1
        """
        def function(*args: SMTTerm) -> SMTTerm:
            substitution = {}
            declarations = []
            term_str = src

            custom_sorts = set()

            for i, arg in enumerate(args, 1):
                smt_sort = arg.get_type()

                if smt_sort.is_custom_type():
                    custom_sorts.add(smt_sort)

                var = FreshSymbol(smt_sort)
                substitution[var] = arg
                declarations.append(f"(declare-fun {var.to_smtlib()} {var.get_type().as_smtlib()})")
                term_str = term_str.replace(f"#{i}", var.to_smtlib())

            # declare uninterpreted sorts
            declarations = [ f"(declare-sort {str(sort)} {sort.arity})" for sort in custom_sorts ] + declarations

            try:
                script_src = " ".join(declarations) + f" (assert {term_str})"
                script = SMTLIB.parse_script(script_src)
            except Exception as e:
                raise Exception(f"unable to parse {script_src}: {e}")

            return script.get_last_formula().substitute(substitution)

        return function
