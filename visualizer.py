from typing import Callable, List

import json
from argparse import ArgumentParser

from dataflow import DataflowGraph, FunctionArgument, ConstantValue


_BINARY_ARITH_PORT_POSITIONS = {
    "dot_input_port_positions": {
        0: "_",
        1: "_",
    },
    "dot_output_port_positions": {
        0: "_",
    },
}


_DOT_CIRCLE_ATTR = "shape=circle fixedsize=true height=0.6 width=0.6"
_DOT_SQUARE_ATTRS = "shape=square fixedsize=true height=0.5 width=0.5"
_DOT_TRIANGLE_ATTRS = "shape=triangle fixedsize=true height=0.6 width=0.513"


OPERATOR_INFO = {
    "MISC_CFG_OP_NOP": {
        "dot_attr": f"label=\"nop\" {_DOT_CIRCLE_ATTR}",
    },
    "ARITH_CFG_OP_ADD": {
        "dot_attr": f"label=\"+\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP2": {
        "dot_attr": f"label=\"+?\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP4": {
        "dot_attr": f"label=\"+?\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SUB": {
        "dot_attr": f"label=\"-\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_MUL": {
        "dot_attr": f"label=\"x\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_CLIP": {
        "dot_attr": f"label=\"Clip\" {_DOT_CIRCLE_ATTR}",
    },
    "ARITH_CFG_OP_SHL": {
        "dot_attr": f"label=\"<<\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ASHR": {
        "dot_attr": f"label=\">>\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_AND": {
        "dot_attr": f"label=\"&\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_OR": {
       "dot_attr": f"label=\"|\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_XOR": {
        "dot_attr": f"label=\"^\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_EQ": {
        "dot_attr": f"label=\"==\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_NE": {
        "dot_attr": f"label=\"!=\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGT": {
        "dot_attr": f"label=\">\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGE": {
        "dot_attr": f"label=\">=\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULT": {
        "dot_attr": f"label=\"<\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULE": {
        "dot_attr": f"label=\"<=\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGT": {
        "dot_attr": f"label=\">\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGE": {
        "dot_attr": f"label=\">=\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLT": {
        "dot_attr": f"label=\"<\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLE": {
        "dot_attr": f"label=\"<=\" {_DOT_CIRCLE_ATTR}",
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "MEM_CFG_OP_LOAD": {
        "dot_attr": f"label=\"LD\" {_DOT_CIRCLE_ATTR} style=filled fillcolor=aquamarine2",
        "dot_input_port_positions": {
            0: "nw",
            1: "n",
            2: "ne",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "MEM_CFG_OP_STORE": {
        "dot_attr": f"label=\"ST\" {_DOT_CIRCLE_ATTR} style=filled fillcolor=aquamarine2",
        "dot_input_port_positions": {
            0: "nw",
            1: "n",
            2: "ne",
            3: "e",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_SELECT": {
        "dot_attr": f"label=\"Sel\" {_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_INVARIANT": {
        "dot_attr": lambda pe: f"label=<Inv<SUB>{'T' if pe.pred == 'CF_CFG_PRED_TRUE' else 'F'}</SUB>> {_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_CARRY": {
        "dot_attr": lambda pe: f"label=<C<SUB>{'T' if pe.pred == 'CF_CFG_PRED_TRUE' else 'F'}</SUB>> {_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
            2: "e",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_MERGE": {
        "dot_attr": f"label=\"M\" {_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
            2: "e",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_ORDER": {
        "dot_attr": f"label=\"O\" {_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
            2: "e",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_STEER": {
        "dot_attr": lambda pe: f"label=\"{'T' if pe.pred == 'CF_CFG_PRED_TRUE' else 'F'}\" {_DOT_TRIANGLE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "STREAM_FU_CFG_T": {
        "dot_attr": lambda pe: f"label=<Str<SUB>{'T' if pe.pred == 'STREAM_CFG_PRED_TRUE' else 'F'}</SUB>> {_DOT_CIRCLE_ATTR} style=filled fillcolor=cadetblue1",
        "dot_input_port_positions": {
            0: "nw",
            1: "n",
            2: "ne",
        },
        "dot_output_port_positions": {
            0: "sw",
            1: "se",
        },
    },
}


class DataflowVisualizer:
    @staticmethod
    def generate_dot_description(graph: DataflowGraph, channel_label: Callable[[int], str] = str) -> str:
        """
        Generate a visualization of the graph in dot format
        """

        elements: List[str] = []
        constant_count = 0

        for pe in graph.vertices:
            attr = OPERATOR_INFO[pe.operator]['dot_attr']
            if callable(attr):
                attr = attr(pe)
            elements.append(f"v{pe.id} [{attr}]")

        for channel in graph.channels:
            if channel.constant is None:
                source_pe = graph.vertices[channel.source]
                dest_pe = graph.vertices[channel.destination]

                # If the input/output port position is specified, we use the specified position
                # otherwise there may be ambiguity, so we mark the edge with the input/output port number
                additional_attributes = ""

                position = OPERATOR_INFO[source_pe.operator].get("dot_output_port_positions", {}).get(channel.source_port, "")
                if position != "":
                    additional_attributes += f" tailport={position}"
                else:
                    additional_attributes += f" taillabel={channel.source_port}"

                position = OPERATOR_INFO[dest_pe.operator].get("dot_input_port_positions", {}).get(channel.destination_port, "")
                if position != "":
                    additional_attributes += f" headport={position}"
                else:
                    additional_attributes += f" headlabel={channel.destination_port}"

                if channel_label is not None:
                    additional_attributes += f" label=\"{channel_label(channel.id)}\""

                elements.append(f"v{channel.source}->v{channel.destination} [arrowsize=0.4{additional_attributes}]")

            else:
                if isinstance(channel.constant, FunctionArgument):
                    label = channel.constant.variable_name                    
                else:
                    assert isinstance(channel.constant, ConstantValue)
                    label = str(channel.constant.value)

                if channel.hold:
                    label += " (hold)"

                elements.append(f"c{constant_count} [label=\"{label}\" shape=plaintext fontsize=7 height=0.1]")

                dest_pe = graph.vertices[channel.destination]
                position = OPERATOR_INFO[dest_pe.operator].get("dot_input_port_positions", {}).get(channel.destination_port, "")
                if position != "":
                    additional_attributes = f"headport={position}"
                else:
                    additional_attributes = f"headlabel={channel.destination_port}"

                if channel_label is not None:
                    additional_attributes += f" label=\"{channel_label(channel.id)}\""

                elements.append(f"c{constant_count}->v{channel.destination} [arrowsize=0.4 {additional_attributes}]")
                constant_count += 1

        return """\
digraph {
  graph [fontname="courier"]
  node [fontname="courier"]
  edge [fontname="courier" fontsize=7]

  """ + "\n  ".join(elements) + "\n}"


def _main():
    parser = ArgumentParser()
    parser.add_argument("o2p", help="Input dataflow graph")
    args = parser.parse_args()

    with open(args.o2p) as dataflow_file:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_file))
        print(DataflowVisualizer.generate_dot_description(dfg))


if __name__ == "__main__":
    _main()
