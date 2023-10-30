from typing import Callable, List, Optional

import json
import html
from argparse import ArgumentParser

from semantics.dataflow.graph import DataflowGraph, FunctionArgument, ConstantValue, ProcessingElement


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
        "dot_label": "nop",
        "dot_attr": _DOT_CIRCLE_ATTR,
    },
    "ARITH_CFG_OP_ID": {
        "dot_label": "id",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ADD": {
        "dot_label": "+",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP": {
        "dot_label": "+?",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP2": {
        "dot_label": "+?",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP4": {
        "dot_label": "+?",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SUB": {
        "dot_label": "-",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_MUL": {
        "dot_label": "x",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_CLIP": {
        "dot_label": "Clip",
        "dot_attr": _DOT_CIRCLE_ATTR,
    },
    "ARITH_CFG_OP_SHL": {
        "dot_label": "<<",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ASHR": {
        "dot_label": ">>",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_AND": {
        "dot_label": "&",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_OR": {
        "dot_label": "|",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_XOR": {
        "dot_label": "^",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_EQ": {
        "dot_label": "==",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_NE": {
        "dot_label": "!=",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGT": {
        "dot_label": "u>",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGE": {
        "dot_label": "u>=",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULT": {
        "dot_label": "u<",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULE": {
        "dot_label": "u<=",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGT": {
        "dot_label": "s>",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGE": {
        "dot_label": "s>=",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLT": {
        "dot_label": "s<",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLE": {
        "dot_label": "s<=",
        "dot_attr": _DOT_CIRCLE_ATTR,
        **_BINARY_ARITH_PORT_POSITIONS,
    },
    "MEM_CFG_OP_LOAD": {
        "dot_label": "LD",
        "dot_attr": f"{_DOT_CIRCLE_ATTR} style=filled fillcolor=aquamarine2",
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
        "dot_label": "ST",
        "dot_attr": f"{_DOT_CIRCLE_ATTR} style=filled fillcolor=aquamarine2",
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
        "dot_label": "Sel",
        "dot_attr": f"{_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_INVARIANT": {
        "dot_label": lambda pe: f"<Inv<FONT POINT-SIZE=\"7\">{'T' if pe.pred == 'CF_CFG_PRED_TRUE' else 'F'}</FONT>>",
        "dot_attr": f"{_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_CARRY": {
        "dot_label": lambda pe: f"<C<FONT POINT-SIZE=\"7\">{'T' if pe.pred == 'CF_CFG_PRED_TRUE' else 'F'}</FONT>>",
        "dot_attr": f"{_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
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
        "dot_label": "M",
        "dot_attr": f"{_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
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
        "dot_label": "O",
        "dot_attr": f"{_DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
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
        "dot_label": lambda pe: "T" if pe.pred == 'CF_CFG_PRED_TRUE' else "F",
        "dot_attr": f"{_DOT_TRIANGLE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "STREAM_FU_CFG_T": {
        "dot_label": lambda pe: f"<Str<FONT POINT-SIZE=\"7\">{'T' if pe.pred == 'STREAM_CFG_PRED_TRUE' else 'F'}</FONT>>",
        "dot_attr": f"{_DOT_CIRCLE_ATTR} style=filled fillcolor=cadetblue1",
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
    def generate_dot_description(
        graph: DataflowGraph,
        pe_label: Optional[Callable[[ProcessingElement], str]] = None,
        channel_label: Optional[Callable[[int], str]] = str,
        llvm_annotation: bool = True,
    ) -> str:
        """
        Generate a visualization of the graph in dot format
        """

        elements: List[str] = []
        constant_count = 0

        for pe in graph.vertices:
            label = OPERATOR_INFO[pe.operator]["dot_label"]
            attr = OPERATOR_INFO[pe.operator]["dot_attr"]
            if callable(label):
                label = label(pe)

            if label.startswith("<") and label.endswith(">"):
                label = label[1:-1]
            else:
                label = html.escape(label)
            
            if pe.llvm_position is not None and llvm_annotation:
                label += f"<FONT point-size=\"5\"><BR/>{pe.llvm_position[0]}:{pe.llvm_position[1]}</FONT>"

            if pe_label is not None:
                label = f"<FONT point-size=\"5\">{pe_label(pe)}<BR/></FONT>" + label

            elements.append(f"v{pe.id} [label=<{label}> {attr}]")

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

                label = "(hold) " if channel.hold else ""

                if channel_label is not None:
                    additional_attributes += f" label=\"{label}{channel_label(channel.id)}\""

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
    parser.add_argument("--no-pe-id", default=False, action="store_const", const=True)
    parser.add_argument("--no-channel-id", default=False, action="store_const", const=True)
    parser.add_argument("--no-llvm-annotation", default=False, action="store_const", const=True)
    args = parser.parse_args()

    with open(args.o2p) as dataflow_file:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_file))
        print(DataflowVisualizer.generate_dot_description(
            dfg,
            None if args.no_pe_id else (lambda pe: str(pe.id)),
            (lambda _: "") if args.no_channel_id else str,
            not args.no_llvm_annotation,
        ))


if __name__ == "__main__":
    _main()
