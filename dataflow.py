from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict, Mapping, Callable
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
                op = "STREAM_FU_CFG_T"

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

#     def generate_dot_description(self: DataflowGraph, channel_label: Callable[[int], str] = str) -> str:
#         """
#         Generate a visualization of the graph in dot format
#         """

#         elements: List[str] = []
#         constant_count = 0

#         for pe in self.vertices:
#             elements.append(f"v{pe.id} [{OPERATOR_INFO[pe.operator]['dot_attr']}]")

#         for channel in self.channels:
#             if channel.constant is None:
#                 source_pe = self.vertices[channel.source]
#                 dest_pe = self.vertices[channel.destination]

#                 # If the input/output port position is specified, we use the specified position
#                 # otherwise there may be ambiguity, so we mark the edge with the input/output port number
#                 additional_attributes = ""

#                 position = OPERATOR_INFO[source_pe.operator].get("dot_output_port_positions", {}).get(channel.source_port, "")
#                 if position != "":
#                     additional_attributes += f" tailport={position}"
#                 else:
#                     additional_attributes += f" taillabel={channel.source_port}"

#                 position = OPERATOR_INFO[dest_pe.operator].get("dot_input_port_positions", {}).get(channel.destination_port, "")
#                 if position != "":
#                     additional_attributes += f" headport={position}"
#                 else:
#                     additional_attributes += f" headlabel={channel.destination_port}"

#                 if channel_label is not None:
#                     additional_attributes += f" label=\"{channel_label(channel.id)}\""

#                 elements.append(f"v{channel.source}->v{channel.destination} [arrowsize=0.4{additional_attributes}]")

#             else:
#                 if isinstance(channel.constant, FunctionArgument):
#                     label = channel.constant.variable_name                    
#                 else:
#                     assert isinstance(channel.constant, ConstantValue)
#                     label = str(channel.constant.value)

#                 if channel.hold:
#                     label += " (hold)"

#                 elements.append(f"c{constant_count} [label=\"{label}\" shape=plaintext fontsize=7 height=0.1]")

#                 dest_pe = self.vertices[channel.destination]
#                 position = OPERATOR_INFO[dest_pe.operator].get("dot_input_port_positions", {}).get(channel.destination_port, "")
#                 if position != "":
#                     additional_attributes = f"headport={position}"
#                 else:
#                     additional_attributes = f"headlabel={channel.destination_port}"

#                 if channel_label is not None:
#                     additional_attributes += f" label=\"{channel_label(channel.id)}\""

#                 elements.append(f"c{constant_count}->v{channel.destination} [arrowsize=0.4 {additional_attributes}]")
#                 constant_count += 1

#         return """\
# digraph {
#   graph [fontname="courier"]
#   node [fontname="courier"]
#   edge [fontname="courier" fontsize=7]

#   """ + "\n  ".join(elements) + "\n}"
