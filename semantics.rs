use vstd::prelude::*;

verus! {

type ChannelIndex = int;
type OperatorIndex = int;
type Value = int;
type State = int;
type Channel = Seq<Value>;

type Address = int;

/** Three types of operators:
 * 1. Normal operators (stateful operators with I/O behaviors determined by the state), no memory access
 * 2. Read operator
 * 3. Store operator
 */

struct Graph {
    operators: Set<OperatorIndex>,
    channels: Set<ChannelIndex>,
    inputs: Map<ChannelIndex, OperatorIndex>,
    outputs: Map<ChannelIndex, OperatorIndex>,
}

enum Operator {
    NonMemory { state: State },
    Read { address: ChannelIndex, sync: ChannelIndex, output: ChannelIndex } ,
    Write { address: ChannelIndex, value: ChannelIndex, sync: ChannelIndex, output: ChannelIndex },
}

struct Configuration {
    operators: Map<OperatorIndex, Operator>,
    channels: Map<ChannelIndex, Channel>,
    memory: Map<Address, Value>,
}

/**
 * These functions are left uninterpreted.
 * For a concrete dataflow graph, one (theoretically) needs to instantiate
 * these functions with the exact behavior.
 * But in our case, we abstract these behaviors out.
 */
spec fn state_inputs(op: OperatorIndex, state: State) -> Seq<ChannelIndex>;
spec fn state_outputs(op: OperatorIndex, state: State) -> Seq<ChannelIndex>;
spec fn state_computation(op: OperatorIndex, state: State, inputs: Seq<Value>) -> (State, Seq<Value>);

spec fn valid_graph(graph: Graph) -> bool
{
    graph.inputs.dom() == graph.channels &&
    graph.outputs.dom() == graph.channels &&
    (forall |channel: ChannelIndex| graph.channels.contains(channel) ==>
        graph.operators.contains(#[trigger] graph.inputs[channel]) &&
        graph.operators.contains(#[trigger] graph.outputs[channel]))
}

/**
 * Conditions for a valid config wrt a dataflow graph
 */
spec fn valid_config(graph: Graph, config: Configuration) -> bool
{
    valid_graph(graph) &&

    config.operators.dom() =~= graph.operators &&
    config.channels.dom() =~= graph.channels &&

    // Inputs and outputs specified by state_inputs and state_outputs should be valid
    // state_computation should behave well regarding input/output lengths
    (forall |op: OperatorIndex, state: State, i: int|
        config.operators.dom().contains(op) &&
        0 <= i < state_inputs(op, state).len() ==>
        graph.outputs[#[trigger] state_inputs(op, state)[i]] == op) &&

    (forall |op: OperatorIndex, state: State, i: int|
        config.operators.dom().contains(op) &&
        0 <= i < state_outputs(op, state).len() ==>
        graph.inputs[#[trigger] state_outputs(op, state)[i]] == op) &&

    (forall |op: OperatorIndex, state: State, inputs: Seq<Value>|
        config.operators.dom().contains(op) &&
        inputs.len() == state_inputs(op, state).len() ==> {
            let (_, outputs) = #[trigger] state_computation(op, state, inputs);
            outputs.len() == state_outputs(op, state).len()
        }) &&

    // For read and write operators, state_* is ignored
    // But their inputs/outputs should still conform to the structure of the graph
    (forall |op: OperatorIndex| config.operators.dom().contains(op) ==> match #[trigger] config.operators[op] {
        Operator::Read { address, sync, output } =>
            graph.outputs[address] == op &&
            graph.outputs[sync] == op &&
            graph.inputs[output] == op,

        Operator::Write { address, value, sync, output } =>
            graph.outputs[address] == op &&
            graph.outputs[value] == op &&
            graph.outputs[sync] == op &&
            graph.inputs[output] == op,

        _ => true,
    })
}

} // verus!
