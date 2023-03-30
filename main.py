import json

import smt

from dataflow import DataflowGraph
from symbolic import SymbolicExecutor


def main():
    with open("parallel.o2p") as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source))

    executor = SymbolicExecutor(dfg, {
        r"array1": smt.FreshSymbol(smt.INT),
        r"array2": smt.FreshSymbol(smt.INT),
        r"len": smt.Int(3), # smt.FreshSymbol(smt.INT),
    })

    config = executor.configurations[0]

    # print(config)

    while True:
        configs = executor.step(config)

        if len(configs) != 1:
            print(configs)
            break
        else:
            config = configs[0]
            print(config)


if __name__ == "__main__":
    main()
