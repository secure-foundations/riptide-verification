from typing import Dict, Tuple, Iterable, List
from dataclasses import dataclass

import json

from argparse import ArgumentParser

import semantics.smt as smt
from semantics.matching import *
from semantics.simulation import SimulationChecker, BackEdgeOnly, IncomingAndBackEdge, LoopHeaderHint

import semantics.dataflow as dataflow
import semantics.llvm as llvm

from utils import logging


logger = logging.getLogger(__name__)


def run_bisim(
    o2p_path: str,
    lso_ll_path: str,
    function_name: Optional[str] = None,
    permission_fractional_reads: int = 4,
    permission_unsat_core: bool = False,
    cut_point_expansion: bool = False,
):
    """
    If function_name is not set, we assume the LLVM module only has one defined function
    """

    with open(o2p_path) as dataflow_source:
        dataflow_graph_json = json.load(dataflow_source)
        dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(dataflow_graph_json)

    with open(lso_ll_path) as llvm_source:
        llvm_module = llvm.Parser.parse_module(llvm_source.read())
        if function_name is None:
            assert len(llvm_module.functions) == 1, f"LLVM module has multiple functions {tuple(llvm_module.functions.keys())}, please specify one to check bisim"
            llvm_function = tuple(llvm_module.functions.values())[0]
        else:
            llvm_function = llvm_module.functions[function_name]

    # Select a cut point placement
    cut_point_placement = BackEdgeOnly(llvm_function, tuple(
        LoopHeaderHint(loop["header"], loop["incoming"], loop["back_edge"])
        for loop in dataflow_graph_json["function"]["loops"]
    ))

    sim_checker = SimulationChecker(
        dataflow_graph,
        llvm_function,
        cut_point_placement,
        permission_fractional_reads=permission_fractional_reads,
        permission_unsat_core=permission_unsat_core,
        cut_point_expansion=cut_point_expansion,
    )
    sim_checker.run_all_checks()


def main():
    parser = ArgumentParser()
    parser.add_argument("o2p", help="Annotated o2p file")
    parser.add_argument("lso_ll", help="LLVM code after lso")
    parser.add_argument("--function", help="Specify a function name to check")
    parser.add_argument("--permission-fractional-reads", default=4, type=int, help="How many read permissions a write permission can be split into")
    parser.add_argument("--permission-unsat-core", action="store_const", const=True, default=False, help="Output unsat core from the permission solver if failed")
    parser.add_argument("--cut-point-expansion", action="store_const", const=True, default=False, help="Enable cut point expansion for confluence checking")
    logging.add_arguments(parser)
    args = parser.parse_args()
    logging.basic_config(args)

    run_bisim(args.o2p, args.lso_ll, args.function, args.permission_fractional_reads, args.permission_unsat_core, args.cut_point_expansion)


if __name__ == "__main__":
    main()

