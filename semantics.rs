use vstd::prelude::*;

verus! {

type ChannelIndex = int;
type OperatorIndex = int;
type Value = int;
type State = int;
type Channel = Seq<Value>;

type Address = int;

spec const WRITE_DEFAULT_OUTPUT: int = 0;

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

spec fn get_op_state(graph: Graph, config: Configuration, op: OperatorIndex) -> State
{
    match config.operators[op] {
        Operator::NonMemory { state } => state,
        Operator::Read { address, sync, output } => 0,
        Operator::Write { address, value, sync, output } => 0,
    }
}

/**
 * Conditions for a valid config wrt a dataflow graph
 */
spec fn valid_config(graph: Graph, config: Configuration) -> bool
{
    valid_graph(graph) &&

    config.operators.dom() =~= graph.operators &&
    config.channels.dom() =~= graph.channels &&

    (forall |addr: Address| config.memory.dom().contains(addr)) &&

    // Inputs and outputs should be different channels
    (forall |op: OperatorIndex, state: State, i: int, j: int|
        graph.operators.contains(op) &&
        0 <= i < j < state_inputs(op, state).len() ==>
        state_inputs(op, state)[i] != state_inputs(op, state)[j]) &&

    (forall |op: OperatorIndex, state: State, i: int, j: int|
        graph.operators.contains(op) &&
        0 <= i < j < state_outputs(op, state).len() ==>
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
            graph.inputs[output] == op &&
            state_inputs(op, get_op_state(graph, config, op)) =~= seq![address, sync] &&
            state_outputs(op, get_op_state(graph, config, op)) =~= seq![output],

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
            graph.inputs[output] == op &&
            state_inputs(op, get_op_state(graph, config, op)) =~= seq![address, value, sync] &&
            state_outputs(op, get_op_state(graph, config, op)) =~= seq![output],

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

            let input_updated_channels = config.channels
                .insert(address, config.channels[address].skip(1))
                .insert(sync, config.channels[sync].skip(1));
            let output_updated_channels = input_updated_channels.insert(output, input_updated_channels[output].push(output_value));

            Configuration {
                operators: config.operators,
                channels: output_updated_channels,
                memory: config.memory,
            }
        },

        Operator::Write { address, value, sync, output } => {
            let address_value = config.channels[address][0];
            let value_value = config.channels[value][0];
            let output_value = WRITE_DEFAULT_OUTPUT;

            let input_updated_channels = config.channels
                .insert(address, config.channels[address].skip(1))
                .insert(value, config.channels[value].skip(1))
                .insert(sync, config.channels[sync].skip(1));
            let output_updated_channels = input_updated_channels.insert(output, input_updated_channels[output].push(output_value));

            Configuration {
                operators: config.operators,
                channels: output_updated_channels,
                memory: config.memory.insert(address_value, value_value),
            }
        },
    }
}

type PermissionAtom = Seq<bool>;
type Permission = Map<Address, PermissionAtom>;
type PermissionAugmentation = Map<ChannelIndex, Seq<Permission>>;

spec fn valid_permission(k_split: int, perm: Permission) -> bool
    recommends k_split > 0
{
    forall |addr: Address| (#[trigger] perm[addr]).len() == k_split
}

/**
 * Condition for two permissions to be considered disjoint
 */
spec fn disjoint_permissions(k_split: int, perm1: Permission, perm2: Permission) -> bool
    recommends
        k_split > 0,
        valid_permission(k_split, perm1),
        valid_permission(k_split, perm2),
{
    forall |addr: Address, i: int| 0 <= i < k_split ==> !(#[trigger] perm1[addr][i] && #[trigger] perm2[addr][i])
}

/**
 * (Disjoint) union of a list of permissions
 */
spec fn union_permissions(k_split: int, perms: Seq<Permission>) -> Permission
    recommends
        k_split > 0,
        perms.len() > 0,
        forall |i: int| 0 <= i < perms.len() ==> valid_permission(k_split, #[trigger] perms[i]),

        // mutually disjoint
        forall |i: int, j: int| 0 <= i < j < perms.len() ==> disjoint_permissions(k_split, perms[i], perms[j]),
{
    Map::new(|addr: Address| true, |addr: Address| Seq::new(perms[0][addr].len(), |i: int| exists |j: int| 0 <= j < perms.len() && #[trigger] perms[j][addr][i]))
}

spec fn has_read_permission(k_split: int, perm: Permission, addr: Address) -> bool
{
    exists |i: int| 0 <= i < k_split && perm[addr][i]
}

spec fn has_write_permission(k_split: int, perm: Permission, addr: Address) -> bool
{
    forall |i: int| 0 <= i < k_split ==> perm[addr][i]
}

/**
 * Condition for perm1 to contain perm2
 */
spec fn contains_permission(k_split: int, perm1: Permission, perm2: Permission) -> bool
    recommends
        k_split > 0,
        valid_permission(k_split, perm1),
        valid_permission(k_split, perm2),
{
    forall |addr: Address, i: int| 0 <= i < perm1[addr].len() ==> (#[trigger] perm2[addr][i] ==> #[trigger] perm1[addr][i])
}

spec fn valid_permission_augmentation(k_split: int, graph: Graph, config: Configuration, aug: PermissionAugmentation) -> bool
    recommends
        k_split > 0,
        valid_config(graph, config),
{
    // Domain should be exactly the set of channels
    (forall |channel: ChannelIndex| aug.dom().contains(channel) ==> #[trigger] graph.channels.contains(channel)) &&

    // Each permission should be valid
    (forall |channel: ChannelIndex| #[trigger] graph.channels.contains(channel) ==>
        aug[channel].len() == config.channels[channel].len() &&
        forall |i: int| 0 <= i < aug[channel].len() ==> valid_permission(k_split, #[trigger] aug[channel][i])) &&

    // Permissions should be mutually disjoint
    (forall |channel1: ChannelIndex, channel2: ChannelIndex, i: int, j: int|
        graph.channels.contains(channel1) && graph.channels.contains(channel2) &&
        0 <= i < aug[channel1].len() && 0 <= j < aug[channel2].len() &&
        (channel1 != channel2 || i != j) ==>
        disjoint_permissions(k_split, aug[channel1][i], aug[channel2][j]))
}

spec fn consistent_step(
    k_split: int,
    graph: Graph,
    op: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
) -> bool
    recommends
        k_split > 0,
        valid_config(graph, config1),
        valid_config(graph, config2),
{
    let inputs = state_inputs(op, get_op_state(graph, config1, op));
    let outputs = state_outputs(op, get_op_state(graph, config1, op));

    let input_perms = Seq::new(inputs.len(), |i: int| aug1[inputs[i]][0]);
    let output_perms = Seq::new(outputs.len(), |i: int| aug2[outputs[i]].last());

    config2 == step(graph, config1, op) &&
    valid_permission_augmentation(k_split, graph, config1, aug1) &&
    valid_permission_augmentation(k_split, graph, config2, aug2) &&

    // Three cases:
    // 1. channel not in inputs or outputs
    // 2. channel in inputs only
    // 3. channel in outputs only
    // 4. channel in both inputs and outputs

    // The permissions are unchanged in case 1
    (forall |channel: ChannelIndex|
        #[trigger] graph.channels.contains(channel) &&
        !inputs.contains(channel) &&
        !outputs.contains(channel) ==>
        aug1[channel] == aug2[channel]
    ) &&

    // Case 2
    (forall |channel: ChannelIndex|
        #[trigger] graph.channels.contains(channel) &&
        inputs.contains(channel) &&
        !outputs.contains(channel) ==>
        aug2[channel].len() == aug1[channel].len() - 1 &&
        aug2[channel] =~= aug1[channel].skip(1)
    ) &&

    // Case 3
    (forall |channel: ChannelIndex|
        #[trigger] graph.channels.contains(channel) &&
        !inputs.contains(channel) &&
        outputs.contains(channel) ==>
        aug2[channel].len() == aug1[channel].len() + 1 &&
        aug2[channel].take(aug2[channel].len() - 1 as int) =~= aug1[channel]
    ) &&

    // Case 4
    (forall |channel: ChannelIndex|
        #[trigger] graph.channels.contains(channel) &&
        inputs.contains(channel) &&
        outputs.contains(channel) ==>
        aug2[channel].len() == aug1[channel].len() &&
        aug2[channel].take(aug2[channel].len() - 1 as int) =~= aug1[channel].skip(1)
    ) &&

    // Union of input perms is less than equal to the union of output perms
    contains_permission(k_split, union_permissions(k_split, input_perms), union_permissions(k_split, output_perms)) &&

    // If the operator is a read/write, we require suitable permissions
    (match config1.operators[op] {
        Operator::Read { address, sync, output } => {
            let address_value = config1.channels[address][0];
            let address_perm = aug1[address][0];
            let sync_perm = aug1[sync][0];
            has_read_permission(k_split, union_permissions(k_split, input_perms), address_value)
        },

        Operator::Write { address, value, sync, output } => {
            let address_value = config1.channels[address][0];
            let address_perm = aug1[address][0];
            let value_perm = aug1[value][0];
            let sync_perm = aug1[sync][0];
            has_write_permission(k_split, union_permissions(k_split, input_perms), address_value)
        },

        _ => true,
    })
}

/**
 * Stepping a valid configuration gives a valid configuration
 */
proof fn lemma_step_valid(graph: Graph, config: Configuration, op: OperatorIndex)
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
proof fn lemma_step_non_memory_commute(graph: Graph, config: Configuration, op1: OperatorIndex, op2: OperatorIndex)
    requires
        valid_config(graph, config),
        fireable(graph, config, op1),
        fireable(graph, config, op2),
        op1 != op2,
        
        is_non_memory(graph, config, op1) ||
        is_non_memory(graph, config, op2) ||
        (config.operators[op1].is_Read() && config.operators[op2].is_Read()),

    ensures
        fireable(graph, step(graph, config, op1), op2),
        fireable(graph, step(graph, config, op2), op1),
        step(graph, step(graph, config, op1), op2) =~~= step(graph, step(graph, config, op2), op1),
{
    let step_1 = step(graph, config, op1);
    let step_2 = step(graph, config, op2);
    let step_1_2 = step(graph, step(graph, config, op1), op2);
    let step_2_1 = step(graph, step(graph, config, op2), op1);

    let op1_init_state = get_op_state(graph, config, op1);
    let op2_init_state = get_op_state(graph, config, op2);

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

/**
 * Lemma: If we have two consistent steps with both operators fireable in the initial config,
 * then their order of execution can be swapped without changing the result.
 */
proof fn lemma_consistent_step_commute(
    k_split: int,
    graph: Graph,
    op1: OperatorIndex, op2: OperatorIndex,
    config1: Configuration, aug1: PermissionAugmentation,
    config2: Configuration, aug2: PermissionAugmentation,
    config3: Configuration, aug3: PermissionAugmentation,
)
    requires
        k_split > 0,
        valid_config(graph, config1),
        valid_config(graph, config2),
        valid_config(graph, config3),

        op1 != op2,

        fireable(graph, config1, op1),
        fireable(graph, config1, op2),

        // config1 -> config2 -> config3 is a consistent trace
        consistent_step(k_split, graph, op1, config1, aug1, config2, aug2),
        consistent_step(k_split, graph, op2, config2, aug2, config3, aug3),

    ensures
        step(graph, step(graph, config1, op2), op1) == config3,
{
    if is_non_memory(graph, config1, op1) || is_non_memory(graph, config1, op2) || (config1.operators[op1].is_Read() && config1.operators[op2].is_Read()){
        lemma_step_non_memory_commute(graph, config1, op1, op2);
    } else {
        let step_1 = step(graph, config1, op1);
        let step_2 = step(graph, config1, op2);
        let step_1_2 = step(graph, step(graph, config1, op1), op2);
        let step_2_1 = step(graph, step(graph, config1, op2), op1);

        let op1_init_state = get_op_state(graph, config1, op1);
        let op2_init_state = get_op_state(graph, config1, op2);

        let op1_inputs = state_inputs(op1, op1_init_state);
        let op1_outputs = state_outputs(op1, op1_init_state);
        let op2_inputs = state_inputs(op2, op2_init_state);
        let op2_outputs = state_outputs(op2, op2_init_state);

        let op1_input_values_in_config = Seq::new(op1_inputs.len(), |i: int| config1.channels[op1_inputs[i]][0]);
        let op1_input_values_in_step_2 = Seq::new(op1_inputs.len(), |i: int| step_2.channels[op1_inputs[i]][0]);
        
        let op2_input_values_in_config = Seq::new(op2_inputs.len(), |i: int| config1.channels[op2_inputs[i]][0]);
        let op2_input_values_in_step_1 = Seq::new(op2_inputs.len(), |i: int| step_1.channels[op2_inputs[i]][0]);
        let op2_input_perms_in_config = Seq::new(op2_inputs.len(), |i: int| aug1[op2_inputs[i]][0]);
        let op2_input_perms_in_step_1 = Seq::new(op2_inputs.len(), |i: int| aug2[op2_inputs[i]][0]);

        assert(op1_input_values_in_config == op1_input_values_in_step_2);
        assert(op2_input_values_in_config == op2_input_values_in_step_1);
        assert(op2_input_perms_in_config == op2_input_perms_in_step_1);
        
        if (config1.operators[op1].is_Read() && config1.operators[op2].is_Write()) {
            // assert(op1_input_values_in_config == op1_input_values_in_step_2);
            // assert(op2_input_values_in_config == op2_input_values_in_step_1);
            // assert(op2_input_perms_in_config == op2_input_perms_in_step_1);
            // assert(state_inputs(op2, get_op_state(graph, config2, op2)) =~= state_inputs(op2, get_op_state(graph, config1, op2)));

            let op1_address = op1_input_values_in_config[0];
            let op2_address = op2_input_values_in_config[0];

            let op1_address_perm = aug1[op1_inputs[0]][0];
            let op1_sync_perm = aug1[op1_inputs[1]][0];
            let op1_input_perm_union = union_permissions(k_split, seq![op1_address_perm, op1_sync_perm]);

            let op2_address_perm = aug1[op2_inputs[0]][0];
            let op2_value_perm = aug1[op2_inputs[1]][0];
            let op2_sync_perm = aug1[op2_inputs[2]][0];
            let op2_input_perm_union = union_permissions(k_split, seq![op2_address_perm, op2_value_perm, op2_sync_perm]);

            assert(seq![op1_address_perm, op1_sync_perm] =~= Seq::new(op1_inputs.len(), |i: int| aug1[op1_inputs[i]][0]));
            assert(seq![op2_address_perm, op2_value_perm, op2_sync_perm] =~= Seq::new(op2_inputs.len(), |i: int| aug1[op2_inputs[i]][0]));

            assert(has_read_permission(k_split, op1_input_perm_union, op1_address));
            assert(has_write_permission(k_split, op2_input_perm_union, op2_address));

            // assume(disjoint_permissions(k_split, op1_input_perm_union, op2_input_perm_union));

            assert(disjoint_permissions(k_split, op1_input_perm_union, op2_input_perm_union)) by {
                assert(disjoint_permissions(k_split, op1_address_perm, op2_address_perm));
                assert(disjoint_permissions(k_split, op1_address_perm, op2_value_perm));
                assert(disjoint_permissions(k_split, op1_address_perm, op2_sync_perm));
                assert(disjoint_permissions(k_split, op1_sync_perm, op2_address_perm));
                assert(disjoint_permissions(k_split, op1_sync_perm, op2_value_perm));
                assert(disjoint_permissions(k_split, op1_sync_perm, op2_sync_perm));

                // assert forall |addr: Address, i: int|
                //     0 <= i < k_split
                // implies
                //     !(op1_input_perm_union[addr][i] && op2_input_perm_union[addr][i]) by
                // {
                //     // if op1_input_perm_union[addr][i] {
                //     //     assert(op1_address_perm[addr][i] || op1_sync_perm[addr][i]);
                //     //     assert(!op2_address_perm[addr][i] && !op2_value_perm[addr][i] && !op2_sync_perm[addr][i]);
                //     // } else {
                //     //     assume(false);
                //     // }
                // }
            }

            assert(op1_address != op2_address) by {
                if op1_address == op2_address {
                    // assert(forall |i: int| 0 <= i < k_split ==> op2_input_perm_union[op1_address][i]);
                    assert(forall |i: int| 0 <= i < k_split ==> !(op2_input_perm_union[op1_address][i] && op1_input_perm_union[op1_address][i]));
                    // assert(exists |i: int| 0 <= i < k_split && op1_input_perm_union[op1_address][i]);
                    assert(false);
                }
            }

            // let op1_output_values_in_step_1 = Seq::new(op1_outputs.len(), |i: int| step_1.channels[op1_outputs[i]][0]);
            // let op1_output_values_in_step_2_1 = Seq::new(op1_outputs.len(), |i: int| step_2_1.channels[op1_outputs[i]][0]);
            // assert(op1_output_values_in_step_1.len() == 1);
            // assert(op1_output_values_in_step_2_1.len() == 1);
            // assert(step_2.memory == config1.memory.insert(op2_address, op2_input_values_in_config[1]));
            // assert(config1.memory[op1_address] == config1.memory.insert(op2_address, op2_input_values_in_config[1])[op1_address]);
        } else if (config1.operators[op1].is_Write() && config1.operators[op2].is_Read()) {
            // assert(op1_input_values_in_config == op1_input_values_in_step_2);
            // assert(op2_input_values_in_config == op2_input_values_in_step_1);
            // assert(op2_input_perms_in_config == op2_input_perms_in_step_1);
            // assert(state_inputs(op2, get_op_state(graph, config2, op2)) =~= state_inputs(op2, get_op_state(graph, config1, op2)));

            let op1_address = op1_input_values_in_config[0];
            let op2_address = op2_input_values_in_config[0];

            let op1_address_perm = aug1[op1_inputs[0]][0];
            let op1_value_perm = aug1[op1_inputs[1]][0];
            let op1_sync_perm = aug1[op1_inputs[2]][0];
            let op1_input_perm_union = union_permissions(k_split, seq![op1_address_perm, op1_value_perm, op1_sync_perm]);

            let op2_address_perm = aug1[op2_inputs[0]][0];
            let op2_sync_perm = aug1[op2_inputs[1]][0];
            let op2_input_perm_union = union_permissions(k_split, seq![op2_address_perm, op2_sync_perm]);

            assert(seq![op1_address_perm, op1_value_perm, op1_sync_perm] =~= Seq::new(op1_inputs.len(), |i: int| aug1[op1_inputs[i]][0]));
            assert(seq![op2_address_perm, op2_sync_perm] =~= Seq::new(op2_inputs.len(), |i: int| aug1[op2_inputs[i]][0]));

            assert(has_write_permission(k_split, op1_input_perm_union, op1_address));
            assert(has_read_permission(k_split, op2_input_perm_union, op2_address));

            assert(disjoint_permissions(k_split, op1_input_perm_union, op2_input_perm_union)) by {
                assert(disjoint_permissions(k_split, op2_address_perm, op1_address_perm));
                assert(disjoint_permissions(k_split, op2_address_perm, op1_value_perm));
                assert(disjoint_permissions(k_split, op2_address_perm, op1_sync_perm));
                assert(disjoint_permissions(k_split, op2_sync_perm, op1_address_perm));
                assert(disjoint_permissions(k_split, op2_sync_perm, op1_value_perm));
                assert(disjoint_permissions(k_split, op2_sync_perm, op1_sync_perm));
            }

            assert(op1_address != op2_address) by {
                if op1_address == op2_address {
                    // assert(forall |i: int| 0 <= i < k_split ==> op2_input_perm_union[op1_address][i]);
                    assert(forall |i: int| 0 <= i < k_split ==> !(op2_input_perm_union[op1_address][i] && op1_input_perm_union[op1_address][i]));
                    // assert(exists |i: int| 0 <= i < k_split && op1_input_perm_union[op1_address][i]);
                    assert(false);
                }
            }

            // assume(false);
            // let op1_output_values_in_step_1 = Seq::new(op1_outputs.len(), |i: int| step_1.channels[op1_outputs[i]][0]);
            // let op1_output_values_in_step_2_1 = Seq::new(op1_outputs.len(), |i: int| step_2_1.channels[op1_outputs[i]][0]);
            // assert(op1_output_values_in_step_1.len() == 1);
            // assert(op1_output_values_in_step_2_1.len() == 1);
            // assert(step_2.memory == config1.memory.insert(op2_address, op2_input_values_in_config[1]));
            // assert(config1.memory[op1_address] == config1.memory.insert(op2_address, op2_input_values_in_config[1])[op1_address]);
        } else if (config1.operators[op1].is_Write() && config1.operators[op2].is_Write()) {
            let op1_address = op1_input_values_in_config[0];
            let op2_address = op2_input_values_in_config[0];

            let op1_address_perm = aug1[op1_inputs[0]][0];
            let op1_value_perm = aug1[op1_inputs[1]][0];
            let op1_sync_perm = aug1[op1_inputs[2]][0];
            let op1_input_perm_union = union_permissions(k_split, seq![op1_address_perm, op1_value_perm, op1_sync_perm]);

            let op2_address_perm = aug1[op2_inputs[0]][0];
            let op2_value_perm = aug1[op2_inputs[1]][0];
            let op2_sync_perm = aug1[op2_inputs[2]][0];
            let op2_input_perm_union = union_permissions(k_split, seq![op2_address_perm, op2_value_perm, op2_sync_perm]);

            assert(seq![op1_address_perm, op1_value_perm, op1_sync_perm] =~= Seq::new(op1_inputs.len(), |i: int| aug1[op1_inputs[i]][0]));
            assert(seq![op2_address_perm, op2_value_perm, op2_sync_perm] =~= Seq::new(op2_inputs.len(), |i: int| aug1[op2_inputs[i]][0]));

            assert(has_write_permission(k_split, op1_input_perm_union, op1_address));
            assert(has_write_permission(k_split, op2_input_perm_union, op2_address));

            assert(disjoint_permissions(k_split, op1_input_perm_union, op2_input_perm_union)) by {
                assert(disjoint_permissions(k_split, op1_address_perm, op2_address_perm));
                assert(disjoint_permissions(k_split, op1_address_perm, op2_value_perm));
                assert(disjoint_permissions(k_split, op1_address_perm, op2_sync_perm));
                assert(disjoint_permissions(k_split, op1_value_perm, op2_address_perm));
                assert(disjoint_permissions(k_split, op1_value_perm, op2_value_perm));
                assert(disjoint_permissions(k_split, op1_value_perm, op2_sync_perm));
                assert(disjoint_permissions(k_split, op1_sync_perm, op2_address_perm));
                assert(disjoint_permissions(k_split, op1_sync_perm, op2_value_perm));
                assert(disjoint_permissions(k_split, op1_sync_perm, op2_sync_perm));
            }

            assert(op1_address != op2_address) by {
                if op1_address == op2_address {
                    // assert(forall |i: int| 0 <= i < k_split ==> op1_input_perm_union[op1_address][i]);
                    // assert(forall |i: int| 0 <= i < k_split ==> op2_input_perm_union[op1_address][i]);
                    // assert(forall |i: int| 0 <= i < k_split ==> !(op2_input_perm_union[op1_address][i] && op1_input_perm_union[op1_address][i]));
                    // // assert(exists |i: int| 0 <= i < k_split && op1_input_perm_union[op1_address][i]);

                    assert forall |i: int|
                        0 <= i < k_split
                    implies
                        op1_input_perm_union[op1_address][i] && false by
                    {
                        assert(op1_input_perm_union[op1_address][i]);
                        assert(op2_input_perm_union[op1_address][i]);
                        assert(!(op1_input_perm_union[op1_address][i] && op2_input_perm_union[op1_address][i]));
                    }

                    let trigger = op1_input_perm_union[op1_address][0];

                    assert(false);
                }
            }
        }

        assert(step_1_2.operators =~= step_2_1.operators);
        assert(step_1_2.memory =~= step_2_1.memory);
        assert(step_1_2.channels =~~= step_2_1.channels);
    }
}

} // verus!
