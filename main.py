import json

from graph import DataflowGraph
from permission import MemoryPermissionSolver


def main():
    with open("test.o2p") as f:
        graph = DataflowGraph.load_dataflow_graph(json.load(f))
        print(graph)
        
        MemoryPermissionSolver.generate_constraints(graph)

        print(graph.generate_dot_description())


if __name__ == "__main__":
    main()
