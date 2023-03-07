from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass


Value = int


class Constant: ...


@dataclass
class FunctionArgument(Constant):
    variable_name: str


@dataclass
class ConstantValue(Constant):
    value: Value


@dataclass
class Channel:
    id: int

    source: Optional[int]
    source_port: Optional[int]
    destination: Optional[int]
    destination_port: Optional[int]

    constant: Optional[Constant]
    hold: bool # whether it holds an infinite number of constant values, only meaning full if constant != None


@dataclass
class ProcessingElement:
    id: int
    name: str
    operator: str
    inputs: Tuple[Channel, ...]
    outputs: Dict[int, Tuple[Channel, ...]]


@dataclass
class DataflowGraph:
    vertices: Tuple[ProcessingElement, ...]
    channels: Tuple[Channel, ...]

    @staticmethod
    def load_dataflow_graph(obj: Any) -> DataflowGraph:
        """
        Load a dataflow graph from an object parsed from the o2p json file (output of the RipTide compiler)
        """

        input_channels: Dict[int, List[Channel]] = {}
        output_channels: Dict[int, Dict[int, List[Channel]]] = {}
        vertices: List[ProcessingElement] = []
        channels: List[Channel] = []

        for i, vertex in enumerate(obj["vertices"]):
            assert i == vertex["ID"]

            input_channels[i] = []

            for port, inputs in enumerate(vertex["inputs"]):
                assert len(inputs) == 1, "unsupported"
                input = inputs[0]

                channel_id = len(channels)

                if input["type"] == "data":
                    input_pe_id = input["ID"]
                    input_pe_port = input["oport"]
                    channel = Channel(channel_id, input_pe_id, input_pe_port, i, port, None, False)
                    
                    if input_pe_id not in output_channels:
                        output_channels[input_pe_id] = {}

                    if input_pe_port not in output_channels[input_pe_id]:
                        output_channels[input_pe_id][input_pe_port] = []

                    output_channels[input_pe_id][input_pe_port].append(channel)

                elif input["type"] == "xdata":
                    channel = Channel(channel_id, None, None, i, port, FunctionArgument(input["name"]), input["hold"])

                elif input["type"] == "const":
                    channel = Channel(channel_id, None, None, i, port, ConstantValue(input["value"]), input["hold"])

                else:
                    assert False, f"unsupported input channel type {input['type']}"

                input_channels[i].append(channel)
                channels.append(channel)

        for i, vertex in enumerate(obj["vertices"]):
            if vertex["type"] == "STREAM_FU_CFG_T":
                op = "STREAM"

            elif vertex["op"] == "CF_CFG_OP_STEER":
                op = "CF_CFG_OP_STEER_TRUE" if vertex["pred"] == "CF_CFG_PRED_TRUE" else "CF_CFG_OP_STEER_FALSE"

            else:
                op = vertex["op"]

            vertices.append(ProcessingElement(
                i, vertex["name"], op,
                tuple(input_channels[i]),
                { k: tuple(v) for k, v in output_channels.get(i, {}).items() }, # convert lists to tuples
            ))

        return DataflowGraph(tuple(vertices), tuple(channels))
