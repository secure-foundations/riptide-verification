"""
Check satisfiability of permission constraints given in a text file
"""

from typing import Tuple, List
from collections import OrderedDict

import argparse

import semantics.dataflow.permission as permission


from lark import Lark, Transformer, Token


class ASTTransformer(Transformer):
    def __init__(self, visit_tokens: bool = True):
        super().__init__(visit_tokens)
        self.heap_objects: OrderedDict[str, None] = OrderedDict()

    def formulas(self, args: List[permission.Formula]) -> Tuple[permission.Formula, ...]:
        return tuple(args)

    def formula(self, args: List[permission.Formula]) -> permission.Formula:
        return args[0]

    def disjunctive_formula(self, args: List[permission.Formula]) -> permission.Disjunction:
        if len(args) == 1:
            return args[0]
        return permission.Disjunction(tuple(args))

    def conjunctive_formula(self, args: List[permission.Formula]) -> permission.Conjunction:
        if len(args) == 1:
            return args[0]
        return permission.Conjunction(tuple(args))

    def atomic_formula(self, args: List[permission.Formula]) -> permission.Formula:
        return args[0]

    def parentheses_formula(self, args: List[permission.Formula]) -> permission.Formula:
        return args[0]

    def inclusion(self, args: List[permission.Term]) -> permission.Inclusion:
        return permission.Inclusion(args[0], args[1])

    def has_read(self, args: List[permission.Term]) -> permission.HasRead:
        self.heap_objects[str(args[0])] = None
        return permission.HasRead(str(args[0]), args[1])

    def equality(self, args: List[permission.Term]) -> permission.Equality:
        return permission.Equality(args[0], args[1])

    def disjoint(self, args: List[permission.Term]) -> permission.Disjoint:
        return permission.Disjoint(tuple(args))

    def disjoint_union(self, args: List[permission.Term]) -> permission.DisjointUnion:
        return permission.DisjointUnion(tuple(args))

    def term(self, args: List[permission.Term]) -> permission.Term:
        return args[0]

    def atomic_term(self, args: List[permission.Term]) -> permission.Term:
        return args[0]

    def variable(self, args: List[Token]) -> permission.Variable:
        return permission.Variable(str(args[0]))

    def read(self, args: List[Token]) -> permission.Read:
        self.heap_objects[str(args[1])] = None
        return permission.Read(str(args[1]), int(str(args[0])))

    def write(self, args: List[Token]) -> permission.Write:
        self.heap_objects[str(args[0])] = None
        return permission.Write(str(args[0]))

    def parentheses_term(self, args: List[permission.Term]) -> permission.Term:
        return args[0]


SYNTAX = r"""
    INLINE_COMMENT: /\#[^\n]*/
    %ignore INLINE_COMMENT
    %ignore /[ \n\t\f\r]+/

    PERMISSION_NAME: /[a-zA-Z][a-zA-Z0-9\-]*/
    HEAP_OBJECT: /%((0|[1-9][0-9]*)|([A-Za-z._][A-Za-z0-9-_'.]*))/
    INTEGER: /0|[1-9][0-9]*/

    formulas: formula*

    formula: disjunctive_formula

    disjunctive_formula: conjunctive_formula ("\\/" conjunctive_formula)*
                       | conjunctive_formula ("or" conjunctive_formula)*

    conjunctive_formula: atomic_formula ("/\\" atomic_formula)*
                       | atomic_formula ("and" atomic_formula)*

    atomic_formula: inclusion
                  | has_read
                  | equality
                  | disjoint
                  | "(" formula ")" -> parentheses_formula

    inclusion: term "<=" term | term "⊑" term

    has_read: "read" HEAP_OBJECT "<=" term | "read" HEAP_OBJECT "⊑" term

    equality: term "=" term

    disjoint: "disjoint" "(" [term ("," term)*] ")"

    term: atomic_term
        | atomic_term ("+" atomic_term)+ -> disjoint_union

    atomic_term: "p" "(" PERMISSION_NAME ")" -> variable
               | "read" "(" INTEGER ")" HEAP_OBJECT -> read
               | "write" HEAP_OBJECT -> write
               | "(" term ")" -> parentheses_term
"""


FORMULAS_PARSER = Lark(
    SYNTAX,
    start="formulas",
    parser="lalr",
    lexer="standard",
    propagate_positions=True,
)


@staticmethod
def parse_formulas(src: str) -> Tuple[Tuple[permission.Formula, ...], Tuple[str, ...]]:
    raw_ast = FORMULAS_PARSER.parse(src)
    transformer = ASTTransformer()
    ast = transformer.transform(raw_ast)
    return ast, tuple(transformer.heap_objects.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file containing a list of permission constraints")
    parser.add_argument("--read-fractions", default=4, type=int, help="How many read permissions can a write permission be split into")
    parser.add_argument("--unsat-core", action="store_const", const=True, default=False, help="Output unsat core")
    parser.add_argument("--solution", action="store_const", const=True, default=False, help="Output solution")
    args = parser.parse_args()

    with open(args.input) as input_file:
        constraints, heap_objects = parse_formulas(input_file.read())

    assert args.read_fractions >= 1
    perm_algebra = permission.FiniteFractionalPA(heap_objects, args.read_fractions)
    result = permission.PermissionSolver.solve_constraints(
        perm_algebra,
        constraints,
        unsat_core=args.unsat_core,
    )

    if isinstance(result, permission.ResultUnsat):
        print("unsat")
    else:
        print("sat")
        if args.solution:
            for var, term in result.solution.items():
                print(f"{var} = {term}")


if __name__ == "__main__":
    main()
