import json

from argparse import ArgumentParser

from graph import DataflowGraph
from permission import *


def examples():
    # satisfiable
    MemoryPermissionSolver.solve_constraints(
        ("A", "B"),
        [
            Inclusion(ReadPermission("A"), PermissionVariable(0)),
            Inclusion(PermissionVariable(0), DisjointUnion.of(
                PermissionVariable(1),
                PermissionVariable(2),
            )),
            Inclusion(PermissionVariable(2), PermissionVariable(0)),
            Inclusion(WritePermission("B"), PermissionVariable(2)),
        ],
    )

    # satisfiable
    MemoryPermissionSolver.solve_constraints(
        ("A", "B"),
        [
            Inclusion(DisjointUnion.of(
                PermissionVariable(0),
                PermissionVariable(1),
            ), PermissionVariable(2)),
            Inclusion(ReadPermission("A"), PermissionVariable(0)),
            Inclusion(ReadPermission("A"), PermissionVariable(1)),
        ],
    )

    # unsatisfiable
    MemoryPermissionSolver.solve_constraints(
        ("A", "B"),
        [
            Inclusion(DisjointUnion.of(
                PermissionVariable(0),
                PermissionVariable(1),
            ), PermissionVariable(2)),
            Inclusion(WritePermission("A"), PermissionVariable(0)),
            Inclusion(ReadPermission("A"), PermissionVariable(1)),
        ],
    )
    
    # unsatisfiable
    MemoryPermissionSolver.solve_constraints(
        ("A", "B"),
        [
            Inclusion(DisjointUnion.of(
                PermissionVariable(0),
                PermissionVariable(1),
            ), PermissionVariable(2)),
            Inclusion(WritePermission("A"), PermissionVariable(0)),
            Inclusion(WritePermission("A"), PermissionVariable(1)),
        ],
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("o2p", help="Dataflow program description (.o2p file)")
    args = parser.parse_args()

    with open(args.o2p) as o2p_file:
        graph = DataflowGraph.load_dataflow_graph(json.load(o2p_file))
        print(graph.generate_dot_description(lambda id: f"p{id}"))
        
        heap_objects, constraints = MemoryPermissionSolver.generate_constraints(graph)

        print("constraints:")
        for constraint in constraints:
            print(f"  {constraint}")

        solution = MemoryPermissionSolver.solve_constraints(heap_objects, constraints)

        if solution is None:
            print("unable to find a solution")
        else:
            for var, term in solution.items():
                print(f"  {var} = {term}")

            print(graph.generate_dot_description(lambda id: f"p{id} = {solution[PermissionVariable(id)]}"))


if __name__ == "__main__":
    main()
    # examples()
