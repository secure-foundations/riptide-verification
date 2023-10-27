from typing import Dict, Tuple, Iterable, List
from dataclasses import dataclass

import json

import semantics.smt as smt
from semantics.matching import *
from semantics.simulation import SimulationChecker, LoopHeaderHint

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def main():
    # with open("examples/test-8/test.test-8.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/test-8/test.test-8.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "for.cond", "for.body",
    #             ("%smax17", "%1", "%inc"),
    #             (),
    #         ),
    #         LoopHeaderHint(
    #             "for.cond2", "for.body5",
    #             ("%smax17", "%lso.alloc.0.lcssa", "%inc8"),
    #             (("%lso.alloc.0", "%lso.alloc.0.lcssa"),),
    #         ),
    #     ]

    with open("examples/test-9/test.test-9.o2p") as dataflow_source:
        dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    with open("examples/test-9/test.test-9.lso.ll") as llvm_source:
        llvm_module = llvm.Parser.parse_module(llvm_source.read())
        llvm_function = tuple(llvm_module.functions.values())[0]
        loop_header_hints = [
            LoopHeaderHint(
                "for.cond", "for.cond.cleanup3",
                ("%smax34", "%lso.alloc.1.lcssa", "%lso.alloc3.1.lcssa", "%inc8"),
                (("%lso.alloc3.1", "%lso.alloc3.1.lcssa"), ("%lso.alloc.1", "%lso.alloc.1.lcssa")),
            ),
            LoopHeaderHint(
                "for.cond1", "for.body4",
                ("%smax34", "%4", "%add", "%inc", "%i.0", "%arrayidx", "%1"),
                (),
            ),
            LoopHeaderHint(
                "for.cond11", "for.body14",
                ("%smax34", "%inc17", "%lso.alloc.0.lcssa"),
                (("%lso.alloc.0", "%lso.alloc.0.lcssa"),),
            ),
        ]

    sim_checker = SimulationChecker(dataflow_graph, llvm_function, loop_header_hints)
    sim_checker.match_llvm_branches()
    sim_checker.generate_dataflow_cut_points()
    sim_checker.check_dataflow_matches()
    sim_checker.check_bisimulation()


if __name__ == "__main__":
    main()

