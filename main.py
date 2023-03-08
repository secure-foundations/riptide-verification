import json

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
    with open("sum.o2p") as f:
        graph = DataflowGraph.load_dataflow_graph(json.load(f))
        print(graph.generate_dot_description())
        
        heap_objects, constraints = MemoryPermissionSolver.generate_constraints(graph)
        for constraint in constraints:
            print(f"  {constraint}")

        solution = MemoryPermissionSolver.solve_constraints(heap_objects, constraints)

        print(graph.generate_dot_description(lambda id: f"p{id} = {solution[PermissionVariable(id)]}"))


if __name__ == "__main__":
    main()
    # examples()
