from typing import Dict, Tuple, Iterable, List
from dataclasses import dataclass

import json

from argparse import ArgumentParser

import semantics.smt as smt
from semantics.matching import *
from semantics.simulation import SimulationChecker, LoopHeaderHint

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def run_bisim(o2p_path: str, lso_ll_path: str, function_name: Optional[str] = None, permission_unsat_core: bool = False):
    """
    If function_name is not set, we assume the LLVM module only has one defined function
    """

    with open(o2p_path) as dataflow_source:
        dataflow_graph_json = json.load(dataflow_source)
        dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(dataflow_graph_json)
        loop_header_hints = [
            LoopHeaderHint(loop["header"], loop["back_edge"], ())
            for loop in dataflow_graph_json["function"]["loops"]
        ]

        for i, loop in enumerate(dataflow_graph_json["function"]["loops"]):
            print(f"llvm cut point {i + 1}: header {loop['header']}, back edge {loop['back_edge']}")

    with open(lso_ll_path) as llvm_source:
        llvm_module = llvm.Parser.parse_module(llvm_source.read())
        if function_name is None:
            assert len(llvm_module.functions) == 1, f"LLVM module has multiple functions {tuple(llvm_module.functions.keys())}, please specify one to check bisim"
            llvm_function = tuple(llvm_module.functions.values())[0]
        else:
            llvm_function = llvm_module.functions[function_name]

    sim_checker = SimulationChecker(dataflow_graph, llvm_function, loop_header_hints, debug=True, permission_unsat_core=permission_unsat_core)
    sim_checker.run_all_checks()


def main():
    parser = ArgumentParser()
    parser.add_argument("o2p", help="Annotated o2p file")
    parser.add_argument("lso_ll", help="LLVM code after lso")
    parser.add_argument("--function", help="Specify a function name to check")
    parser.add_argument("--permission-unsat-core", action="store_const", const=True, default=False, help="Output unsat core from the permission solver if failed")
    args = parser.parse_args()

    run_bisim(args.o2p, args.lso_ll, args.function, args.permission_unsat_core)


if __name__ == "__main__":
    main()

