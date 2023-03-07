BINARY_ARITH_PORT_POSITIONS = {
    "dot_input_port_positions": {
        0: "_",
        1: "_",
    },
    "dot_output_port_positions": {
        0: "_",
    },
}


DOT_CIRCLE_ATTR = "shape=circle fixedsize=true height=0.5 width=0.5"
DOT_SQUARE_ATTRS = "shape=square fixedsize=true height=0.45 width=0.45"
DOT_TRIANGLE_ATTRS = "shape=triangle fixedsize=true height=0.6 width=0.513"


OPERATOR_INFO = {
    "MISC_CFG_OP_NOP": {
        "dot_attr": f"label=\"nop\" {DOT_CIRCLE_ATTR}",
    },
    "ARITH_CFG_OP_ADD": {
        "dot_attr": f"label=\"+\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP2": {
        "dot_attr": f"label=\"+?\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP4": {
        "dot_attr": f"label=\"+?\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SUB": {
        "dot_attr": f"label=\"-\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_MUL": {
        "dot_attr": f"label=\"x\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_CLIP": {
        "dot_attr": f"label=\"Clip\" {DOT_CIRCLE_ATTR}",
    },
    "ARITH_CFG_OP_SHL": {
        "dot_attr": f"label=\"<<\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ASHR": {
        "dot_attr": f"label=\">>\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_AND": {
        "dot_attr": f"label=\"&\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_OR": {
       "dot_attr": f"label=\"|\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_XOR": {
        "dot_attr": f"label=\"^\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_EQ": {
        "dot_attr": f"label=\"==\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_NE": {
        "dot_attr": f"label=\"!=\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGT": {
        "dot_attr": f"label=\">\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGE": {
        "dot_attr": f"label=\">=\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULT": {
        "dot_attr": f"label=\"<\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULE": {
        "dot_attr": f"label=\"<=\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGT": {
        "dot_attr": f"label=\">\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGE": {
        "dot_attr": f"label=\">=\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLT": {
        "dot_attr": f"label=\"<\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLE": {
        "dot_attr": f"label=\"<=\" {DOT_CIRCLE_ATTR}",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "MEM_CFG_OP_LOAD": {
        "dot_attr": f"label=\"LD\" {DOT_CIRCLE_ATTR} style=filled fillcolor=aquamarine2",
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
        "dot_attr": f"label=\"ST\" {DOT_CIRCLE_ATTR} style=filled fillcolor=aquamarine2",
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
        "dot_attr": f"label=\"Sel\" {DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_INVARIANT": {
        "dot_attr": f"label=\"Inv\" {DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_CARRY": {
        "dot_attr": f"label=\"C\" {DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
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
        "dot_attr": f"label=\"M\" {DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
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
        "dot_attr": f"label=\"O\" {DOT_SQUARE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
            2: "e",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_STEER_TRUE": {
        "dot_attr": f"label=\"T\" {DOT_TRIANGLE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_STEER_FALSE": {
        "dot_attr": f"label=\"F\" {DOT_TRIANGLE_ATTRS} style=filled fillcolor=grey",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "STREAM_FU_CFG_T": {
        "dot_attr": f"label=\"Str\" {DOT_CIRCLE_ATTR} style=filled fillcolor=cadetblue1",
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
