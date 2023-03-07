BINARY_ARITH_PORT_POSITIONS = {
    "dot_input_port_positions": {
        0: "_",
        1: "_",
    },
    "dot_output_port_positions": {
        0: "_",
    },
}


OPERATOR_INFO = {
    "MISC_CFG_OP_NOP": {
        "dot_attr": "label=\"nop\" shape=circle fixedsize=true height=0.5 width=0.5",
    },
    "ARITH_CFG_OP_ADD": {
        "dot_attr": "label=\"+\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP2": {
        "dot_attr": "label=\"+?\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_GEP4": {
        "dot_attr": "label=\"+?\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SUB": {
        "dot_attr": "label=\"-\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_MUL": {
        "dot_attr": "label=\"x\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "MUL_CFG_OP_CLIP": {
        "dot_attr": "label=\"Clip\" shape=circle fixedsize=true height=0.5 width=0.5",
    },
    "ARITH_CFG_OP_SHL": {
        "dot_attr": "label=\"<<\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ASHR": {
        "dot_attr": "label=\">>\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_AND": {
        "dot_attr": "label=\"&\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_OR": {
       "dot_attr": "label=\"|\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_XOR": {
        "dot_attr": "label=\"^\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_EQ": {
        "dot_attr": "label=\"==\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_NE": {
        "dot_attr": "label=\"!=\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGT": {
        "dot_attr": "label=\">\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_UGE": {
        "dot_attr": "label=\">=\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULT": {
        "dot_attr": "label=\"<\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_ULE": {
        "dot_attr": "label=\"<=\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGT": {
        "dot_attr": "label=\">\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SGE": {
        "dot_attr": "label=\">=\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLT": {
        "dot_attr": "label=\"<\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "ARITH_CFG_OP_SLE": {
        "dot_attr": "label=\"<=\" shape=circle fixedsize=true height=0.5 width=0.5",
        **BINARY_ARITH_PORT_POSITIONS,
    },
    "MEM_CFG_OP_LOAD": {
        "dot_attr": "label=\"LD\" shape=circle fixedsize=true height=0.5 width=0.5",
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
        "dot_attr": "label=\"ST\" shape=circle fixedsize=true height=0.5 width=0.5",
        "dot_input_port_positions": {
            0: "nw",
            1: "n",
            2: "ne",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_SELECT": {
        "dot_attr": "label=\"Sel\" shape=circle fixedsize=true height=0.45 width=0.45",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_INVARIANT": {
        "dot_attr": "label=\"Inv\" shape=circle fixedsize=true height=0.45 width=0.45",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_CARRY": {
        "dot_attr": "label=\"C\" shape=square fixedsize=true height=0.45 width=0.45",
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
        "dot_attr": "label=\"M\" shape=square fixedsize=true height=0.45 width=0.45",
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
        "dot_attr": "label=\"O\" shape=square fixedsize=true height=0.45 width=0.45",
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
        "dot_attr": "label=\"T\" shape=triangle fixedsize=true height=0.6 width=0.513",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "CF_CFG_OP_STEER_FALSE": {
        "dot_attr": "label=\"F\" shape=triangle fixedsize=true height=0.6 width=0.513",
        "dot_input_port_positions": {
            0: "w",
            1: "n",
        },
        "dot_output_port_positions": {
            0: "s",
        },
    },
    "STREAM_FU_CFG_T": {
        "dot_attr": "label=\"Str\" shape=circle fixedsize=true height=0.5 width=0.5",
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
