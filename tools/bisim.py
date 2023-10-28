from typing import Dict, Tuple, Iterable, List
from dataclasses import dataclass

import json

import semantics.smt as smt
from semantics.matching import *
from semantics.simulation import SimulationChecker, LoopHeaderHint

import semantics.dataflow as dataflow
import semantics.llvm as llvm


def main():
    # with open("examples/test-7/test.test-7.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/test-7/test.test-7.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "for.cond", "for.inc",
    #             (),
    #         ),
    #     ]

    # with open("examples/test-8/test.test-8.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/test-8/test.test-8.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "for.cond", "for.body",
    #             (),
    #         ),
    #         LoopHeaderHint(
    #             "for.cond2", "for.body5",
    #             (("%lso.alloc.0", "%lso.alloc.0.lcssa"),),
    #         ),
    #     ]

    # with open("examples/test-9/test.test-9.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/test-9/test.test-9.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "for.cond", "for.cond.cleanup3",
    #             (("%lso.alloc3.1", "%lso.alloc3.1.lcssa"), ("%lso.alloc.1", "%lso.alloc.1.lcssa")),
    #         ),
    #         LoopHeaderHint(
    #             "for.cond1", "for.body4",
    #             (),
    #         ),
    #         LoopHeaderHint(
    #             "for.cond11", "for.body14",
    #             (("%lso.alloc.0", "%lso.alloc.0.lcssa"),),
    #         ),
    #     ]

    # TODO: doesn't work
    # with open("examples/bfs/bfs.bfs.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/bfs/bfs.bfs.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "while.cond", "while.cond.loopexit",
    #             (),
    #         ),
    #         LoopHeaderHint(
    #             "for.cond", "if.end",
    #             (),
    #         ),
    #     ]

    with open("examples/dmm/dmm.dmm.o2p") as dataflow_source:
        dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    with open("examples/dmm/dmm.dmm.lso.ll") as llvm_source:
        llvm_module = llvm.Parser.parse_module(llvm_source.read())
        llvm_function = tuple(llvm_module.functions.values())[0]
        loop_header_hints = [
            LoopHeaderHint(
                "for.cond", "for.cond.cleanup3",
                (("%dest_idx.1", "%dest_idx.1.lcssa"),),
            ),
            LoopHeaderHint(
                "for.cond1", "for.cond.cleanup7",
                (),
            ),
            LoopHeaderHint(
                "for.cond5", "for.body8",
                (),
            ),
        ]

    # with open("examples/fft_ns0/fft_ns0.fft_ns0.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/fft_ns0/fft_ns0.fft_ns0.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "for.cond", "for.body",
    #             (),
    #         ),
    #     ]

    # with open("examples/test-11/test.test-11.o2p") as dataflow_source:
    #     dataflow_graph = dataflow.DataflowGraph.load_dataflow_graph(json.load(dataflow_source))
        
    # with open("examples/test-11/test.test-11.lso.ll") as llvm_source:
    #     llvm_module = llvm.Parser.parse_module(llvm_source.read())
    #     llvm_function = tuple(llvm_module.functions.values())[0]
    #     loop_header_hints = [
    #         LoopHeaderHint(
    #             "for.cond", "for.cond.cleanup3",
    #             (("%lso.alloc1.1", "%lso.alloc1.1.lcssa"),),
    #         ),
    #         LoopHeaderHint(
    #             "for.cond1", "for.body4",
    #             (),
    #         ),
    #     ]

    sim_checker = SimulationChecker(dataflow_graph, llvm_function, loop_header_hints)
    sim_checker.match_llvm_branches()
    sim_checker.generate_dataflow_cut_points()
    sim_checker.check_dataflow_matches()
    sim_checker.check_bisimulation()


if __name__ == "__main__":
    main()
