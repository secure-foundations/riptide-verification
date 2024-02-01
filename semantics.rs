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

#[is_variant]
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

    // Inputs and outputs should be different channels
    (forall |op: OperatorIndex, state: State, i: int, j: int|
        graph.operators.contains(op) &&
        0 <= i < state_inputs(op, state).len() &&
        0 <= j < state_inputs(op, state).len() &&
        i != j ==>
        state_inputs(op, state)[i] != state_inputs(op, state)[j]) &&

    (forall |op: OperatorIndex, state: State, i: int, j: int|
        graph.operators.contains(op) &&
        0 <= i < state_outputs(op, state).len() &&
        0 <= j < state_outputs(op, state).len() &&
        i != j ==>
        state_outputs(op, state)[i] != state_outputs(op, state)[j]) &&

    // Inputs and outputs are valid channel indices
    (forall |op: OperatorIndex, state: State, i: int|
        graph.operators.contains(op) &&
        0 <= i < state_inputs(op, state).len() ==>
        graph.channels.contains(#[trigger] state_inputs(op, state)[i])) &&

    (forall |op: OperatorIndex, state: State, i: int|
        graph.operators.contains(op) &&
        0 <= i < state_outputs(op, state).len() ==>
        graph.channels.contains(#[trigger] state_outputs(op, state)[i])) &&

    // Inputs and outputs specified by state_inputs and state_outputs should be valid
    // state_computation should behave well regarding input/output lengths
    (forall |op: OperatorIndex, state: State, i: int|
        graph.operators.contains(op) &&
        0 <= i < state_inputs(op, state).len() ==>
        graph.outputs[#[trigger] state_inputs(op, state)[i]] == op) &&

    (forall |op: OperatorIndex, state: State, i: int|
        graph.operators.contains(op) &&
        0 <= i < state_outputs(op, state).len() ==>
        graph.inputs[#[trigger] state_outputs(op, state)[i]] == op) &&

    (forall |op: OperatorIndex, state: State, inputs: Seq<Value>|
        graph.operators.contains(op) &&
        inputs.len() == state_inputs(op, state).len() ==> {
            let (_, outputs) = #[trigger] state_computation(op, state, inputs);
            outputs.len() == state_outputs(op, state).len()
        }) &&

    // For read and write operators, state_* is ignored
    // But their inputs/outputs should still conform to the structure of the graph
    (forall |op: OperatorIndex| graph.operators.contains(op) ==> match #[trigger] config.operators[op] {
        Operator::Read { address, sync, output } =>
            address != sync &&
            graph.channels.contains(address) &&
            graph.channels.contains(sync) &&
            graph.channels.contains(output) &&
            graph.outputs[address] == op &&
            graph.outputs[sync] == op &&
            graph.inputs[output] == op,

        Operator::Write { address, value, sync, output } =>
            address != value &&
            address != sync &&
            value != sync &&
            graph.channels.contains(address) &&
            graph.channels.contains(value) &&
            graph.channels.contains(sync) &&
            graph.channels.contains(output) &&
            graph.outputs[address] == op &&
            graph.outputs[value] == op &&
            graph.outputs[sync] == op &&
            graph.inputs[output] == op,

        _ => true,
    })
}

/**
 * Condition for an operator to be fireable in a configuration
 */
spec fn fireable(graph: Graph, config: Configuration, op: OperatorIndex) -> bool
    recommends valid_config(graph, config)
{
    graph.operators.contains(op) &&
    match config.operators[op] {
        Operator::NonMemory { state } =>
            (forall |channel: ChannelIndex|
                #[trigger] state_inputs(op, state).contains(channel) ==>
                config.channels[channel].len() > 0),
        
        Operator::Read { address, sync, output } =>
            config.channels[address].len() > 0 &&
            config.channels[sync].len() > 0,
        
        Operator::Write { address, value, sync, output } =>
            config.channels[address].len() > 0 &&
            config.channels[value].len() > 0 &&
            config.channels[sync].len() > 0,
    }
}

spec fn is_non_memory(graph: Graph, config: Configuration, op: OperatorIndex) -> bool
    recommends
        valid_config(graph, config),
        graph.operators.contains(op),
{
    match config.operators[op] {
        Operator::NonMemory { state } => true,
        _ => false,
    }
}

/**
 * Fire an operator in a given configuration
 */
spec fn step(graph: Graph, config: Configuration, op: OperatorIndex) -> Configuration
    recommends
        valid_config(graph, config),
        fireable(graph, config, op),
{
    match config.operators[op] {
        Operator::NonMemory { state } => {
            let input_channels = state_inputs(op, state);
            let output_channels = state_outputs(op, state);
            let inputs = Seq::new(input_channels.len(), |i: int| config.channels[input_channels[i]][0]);
            let (next_state, outputs) = state_computation(op, state, inputs);

            // Pop input channels first
            let input_updated_channels = Map::new(|channel: ChannelIndex| graph.channels.contains(channel), |channel: ChannelIndex|
                if input_channels.contains(channel) { config.channels[channel].skip(1) }
                else { config.channels[channel] }
            );

            // Then push to output channels
            let output_updated_channels = Map::new(|channel: ChannelIndex| graph.channels.contains(channel), |channel: ChannelIndex|
                if output_channels.contains(channel) {
                    let output_index = output_channels.index_of(channel);
                    input_updated_channels[channel].push(outputs[output_index])
                }
                else { input_updated_channels[channel] }
            );

            Configuration {
                operators: config.operators.insert(op, Operator::NonMemory { state: next_state }),
                channels: output_updated_channels,
                memory: config.memory,
            }
        },

        Operator::Read { address, sync, output } => {
            let address_value = config.channels[address][0];
            let output_value = config.memory[address_value];

            let input_updated_channels1 = config.channels.insert(address, config.channels[address].take(1));
            let input_updated_channels2 = input_updated_channels1.insert(sync, input_updated_channels1[sync].take(1));
            let output_updated_channels = input_updated_channels2.insert(output, input_updated_channels2[output].push(output_value));

            Configuration {
                operators: config.operators,
                channels: output_updated_channels,
                memory: config.memory,
            }
        },

        Operator::Write { address, value, sync, output } => {
            let address_value = config.channels[address][0];
            let value_value = config.channels[value][0];
            let output_value = 1;

            let input_updated_channels1 = config.channels.insert(address, config.channels[address].take(1));
            let input_updated_channels2 = input_updated_channels1.insert(value, input_updated_channels1[value].take(1));
            let input_updated_channels3 = input_updated_channels2.insert(sync, input_updated_channels2[sync].take(1));
            let output_updated_channels = input_updated_channels3.insert(output, input_updated_channels3[output].push(output_value));

            Configuration {
                operators: config.operators,
                channels: output_updated_channels,
                memory: config.memory.insert(address_value, value_value),
            }
        },
    }
}

/**
 * Stepping a valid configuration gives a valid configuration
 */
proof fn step_valid(graph: Graph, config: Configuration, op: OperatorIndex)
    requires
        valid_config(graph, config),
        fireable(graph, config, op),

    ensures
        valid_config(graph, step(graph, config, op)),
{
    // match config.operators[op] {
    //     Operator::NonMemory { state } => {},
    //     Operator::Read { address, sync, output } => {},
    //     Operator::Write { address, value, sync, output } => {},
    // }
}

/**
 * Lemma: For two different and non-memor operators op1 and op2,
 * if both of them are fireable in a configuration, then their execution
 * is commutable.
 * 
 * i.e. without memory operators, dataflow graph execution is locally confluent.
 */
proof fn step_non_memory_commute(graph: Graph, config: Configuration, op1: OperatorIndex, op2: OperatorIndex)
    requires
        valid_config(graph, config),
        fireable(graph, config, op1),
        fireable(graph, config, op2),
        op1 != op2,
        is_non_memory(graph, config, op1),
        is_non_memory(graph, config, op2),

    ensures
        fireable(graph, step(graph, config, op1), op2),
        fireable(graph, step(graph, config, op2), op1),
        step(graph, step(graph, config, op1), op2) =~~= step(graph, step(graph, config, op2), op1),
{
    let step_1 = step(graph, config, op1);
    let step_2 = step(graph, config, op2);
    let step_1_2 = step(graph, step(graph, config, op1), op2);
    let step_2_1 = step(graph, step(graph, config, op2), op1);

    let op1_init_state = config.operators[op1].get_NonMemory_state();
    let op2_init_state = config.operators[op2].get_NonMemory_state();

    let op1_inputs = state_inputs(op1, op1_init_state);
    let op1_outputs = state_outputs(op1, op1_init_state);
    let op2_inputs = state_inputs(op2, op2_init_state);
    let op2_outputs = state_outputs(op2, op2_init_state);

    // assert(forall |channel: ChannelIndex| #[trigger] op1_inputs.contains(channel) ==>
    //     #[trigger] step_2.channels[channel][0] == #[trigger] config.channels[channel][0]);
    // assert(forall |channel: ChannelIndex| #[trigger] op2_inputs.contains(channel) ==>
    //     #[trigger] step_1.channels[channel][0] == #[trigger] config.channels[channel][0]);

    // Step in op2 does not alter the input values to op1
    let op1_input_values_in_config = Seq::new(op1_inputs.len(), |i: int| config.channels[op1_inputs[i]][0]);
    let op1_input_values_in_step_2 = Seq::new(op1_inputs.len(), |i: int| step_2.channels[op1_inputs[i]][0]);

    assert forall |i: int| 0 <= i < op1_inputs.len() implies op1_input_values_in_config[i] == op1_input_values_in_step_2[i] by {
        assert(op1_inputs.contains(op1_inputs[i]));
    }
    assert(op1_input_values_in_config == op1_input_values_in_step_2);

    // Step in op1 does not alter the input values to op2
    let op2_input_values_in_config = Seq::new(op2_inputs.len(), |i: int| config.channels[op2_inputs[i]][0]);
    let op2_input_values_in_step_1 = Seq::new(op2_inputs.len(), |i: int| step_1.channels[op2_inputs[i]][0]);

    assert forall |i: int| 0 <= i < op2_inputs.len() implies op2_input_values_in_config[i] == op2_input_values_in_step_1[i] by {
        assert(op2_inputs.contains(op2_inputs[i]));
    }
    assert(op2_input_values_in_config == op2_input_values_in_step_1);

    assert(step_1_2.operators =~= step_2_1.operators);
    assert(step_1_2.channels =~~= step_2_1.channels);
}

} // verus!
