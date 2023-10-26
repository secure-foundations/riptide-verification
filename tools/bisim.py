from typing import Dict, Tuple, Iterable, List
from dataclasses import dataclass

import json

import semantics.smt as smt
from semantics.matching import *
from semantics.simulation import SimulationChecker, LoopHeaderHint

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def main():
    with open("examples/test-4/test-4.o2p") as dataflow_source:
        dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    with open("examples/test-4/test-4.lso.ll") as llvm_source:
        llvm_module = llvm.Parser.parse_module(llvm_source.read())
        llvm_function = tuple(llvm_module.functions.values())[0]
        loop_header_hints = [
            LoopHeaderHint(
                "for.cond", "for.cond.cleanup3",
                ("%smax", "%lso.alloc2.1.lcssa", "%inc8"),
                (("%lso.alloc2.1", "%lso.alloc2.1.lcssa"),),
            ),
            LoopHeaderHint(
                "for.cond1", "for.body4",
                ("%smax", "%4", "%add", "%inc", "%i.0", "%1", "%arrayidx"),
                (),
            ),
        ]

    sim_checker = SimulationChecker(dataflow_graph, llvm_function, loop_header_hints)
    sim_checker.match_llvm_branches()
    sim_checker.generate_dataflow_cut_points()
    sim_checker.check_dataflow_matches()
    sim_checker.check_bisimulation()


if __name__ == "__main__":
    main()

