use vstd::prelude::*;

verus! {

pub struct ChannelIndex { v: int }
pub struct OperatorIndex { v: int }
pub struct Value { v: int }
pub struct Address { v: int }
pub struct State { v: int }

impl Value {
    pub spec fn as_address(self) -> Address;
}

pub spec fn write_default_output() -> Value;
pub spec fn read_write_default_state() -> State;

pub struct Graph {
    pub operators: Set<OperatorIndex>,
    pub channels: Set<ChannelIndex>,
    pub inputs: Map<ChannelIndex, OperatorIndex>,
    pub outputs: Map<ChannelIndex, OperatorIndex>,
}

#[is_variant]
pub enum Operator {
    NonMemory { state: State }, // operators w/o memory access (I/O behaviors determined by the state)
    Read { address: ChannelIndex, sync: ChannelIndex, output: ChannelIndex } ,
    Write { address: ChannelIndex, value: ChannelIndex, sync: ChannelIndex, output: ChannelIndex },
}

pub struct Configuration {
    pub graph: Graph,
    pub operators: Map<OperatorIndex, Operator>,
    pub channels: Map<ChannelIndex, Seq<Value>>,
    pub memory: Map<Address, Value>,
}

/**
 * These functions are left uninterpreted.
 * For a concrete dataflow graph, one (theoretically) needs to instantiate
 * these functions with the exact behavior.
 * But in our case, we abstract these behaviors out.
 */
pub spec fn state_inputs(op: OperatorIndex, state: State) -> Seq<ChannelIndex>;
pub spec fn state_outputs(op: OperatorIndex, state: State) -> Seq<ChannelIndex>;
pub spec fn state_computation(op: OperatorIndex, state: State, inputs: Seq<Value>) -> (State, Seq<Value>);

/**
 * A bound on the size of all channels. 0 means infinite
 */
pub spec fn channel_bound() -> nat;

impl Graph {
    pub open spec fn valid(self) -> bool
    {
        self.inputs.dom() == self.channels &&
        self.outputs.dom() == self.channels &&
        (forall |channel: ChannelIndex|
            #![trigger self.inputs[channel]]
            #![trigger self.outputs[channel]]
            self.channels.contains(channel) ==>
            self.operators.contains(self.inputs[channel]) &&
            self.operators.contains(self.outputs[channel]))
    }
}

impl Configuration {
    pub open spec fn is_op(self, op: OperatorIndex) -> bool
    {
        self.graph.operators.contains(op)
    }

    pub open spec fn is_channel(self, channel: ChannelIndex) -> bool
    {
        self.graph.channels.contains(channel)
    }

    pub open spec fn get_op_state(self, op: OperatorIndex) -> State
        recommends
            self.is_op(op)
    {
        match self.operators[op] {
            Operator::NonMemory { state } => state,
            Operator::Read { address, sync, output } => read_write_default_state(),
            Operator::Write { address, value, sync, output } => read_write_default_state(),
        }
    }

    /**
     * Getting the input channels of an operator
     */
    pub open spec fn get_op_input_channels(self, op: OperatorIndex) -> Seq<ChannelIndex>
    {
        state_inputs(op, self.get_op_state(op))
    }

    /**
     * Getting the output channels of an operator
     */
    pub open spec fn get_op_output_channels(self, op: OperatorIndex) -> Seq<ChannelIndex>
    {
        state_outputs(op, self.get_op_state(op))
    }

    /**
     * Conditions for a valid config wrt a dataflow graph
     */
    pub open spec fn valid(self) -> bool
    {
        self.graph.valid() &&

        self.operators.dom() =~= self.graph.operators &&
        self.channels.dom() =~= self.graph.channels &&

        // Memory map is total
        (forall |addr: Address| self.memory.dom().contains(addr)) &&

        // Inputs and outputs should be different channels
        (forall |op: OperatorIndex, state: State|
            self.is_op(op) ==> (#[trigger] state_inputs(op, state)).no_duplicates()) &&

        (forall |op: OperatorIndex, state: State|
            self.is_op(op) ==> (#[trigger] state_outputs(op, state)).no_duplicates()) &&

        // Inputs and outputs are valid channels
        (forall |op: OperatorIndex, state: State, i: int|
            self.is_op(op) &&
            0 <= i < state_inputs(op, state).len() ==>
            self.is_channel(#[trigger] state_inputs(op, state)[i])) &&

        (forall |op: OperatorIndex, state: State, i: int|
            self.is_op(op) &&
            0 <= i < state_outputs(op, state).len() ==>
            self.is_channel(#[trigger] state_outputs(op, state)[i])) &&

        // Inputs and outputs specified by state_inputs and state_outputs should be valid
        // state_computation should behave well regarding input/output lengths
        (forall |op: OperatorIndex, state: State, i: int|
            self.is_op(op) &&
            0 <= i < state_inputs(op, state).len() ==>
            self.graph.outputs[#[trigger] state_inputs(op, state)[i]] == op) &&

        (forall |op: OperatorIndex, state: State, i: int|
            self.is_op(op) &&
            0 <= i < state_outputs(op, state).len() ==>
            self.graph.inputs[#[trigger] state_outputs(op, state)[i]] == op) &&

        // Output values of state_computation should have matching length in state_outputs
        (forall |op: OperatorIndex, state: State, inputs: Seq<Value>|
            self.is_op(op) &&
            inputs.len() == state_inputs(op, state).len() ==> {
                let (_, outputs) = #[trigger] state_computation(op, state, inputs);
                outputs.len() == state_outputs(op, state).len()
            }) &&

        // For read and write operators, state_* is ignored
        // But their inputs/outputs should still conform to the structure of the graph
        (forall |op: OperatorIndex| self.is_op(op) ==> match #[trigger] self.operators[op] {
            Operator::Read { address, sync, output } =>
                self.get_op_input_channels(op) =~= seq![address, sync] &&
                self.get_op_output_channels(op) =~= seq![output],

            Operator::Write { address, value, sync, output } =>
                self.get_op_input_channels(op) =~= seq![address, value, sync] &&
                self.get_op_output_channels(op) =~= seq![output],

            _ => true,
        }) &&

        // Channels should be properly bounded
        if channel_bound() > 0 {
            forall |channel: ChannelIndex| #[trigger] self.is_channel(channel) ==> self.channels[channel].len() <= channel_bound()
        } else {
            true
        }
    }

    /**
     * Condition for an operator to be fireable in a configuration
     */
    pub open spec fn fireable(self, op: OperatorIndex) -> bool
        recommends self.valid()
    {
        self.graph.operators.contains(op) &&

        // Available values in input channels
        (forall |channel: ChannelIndex|
            #[trigger] self.is_channel(channel) &&
            self.get_op_input_channels(op).contains(channel)
            ==>
            self.channels[channel].len() > 0) &&

        // Available space in output channels (if the output is not one of the input channels)
        if channel_bound() > 0 {
            forall |channel: ChannelIndex|
                #[trigger] self.is_channel(channel) &&
                !self.get_op_input_channels(op).contains(channel) &&
                self.get_op_output_channels(op).contains(channel)
                ==>
                self.channels[channel].len() < channel_bound()
        } else {
            true
        }
    }

    /**
     * Getting the input values of an operator (if it's fireable)
     */
    pub open spec fn get_op_input_values(self, op: OperatorIndex) -> Seq<Value>
        recommends
            self.valid(),
            self.fireable(op),
    {
        let input_channels = self.get_op_input_channels(op);
        Seq::new(input_channels.len(), |i: int| self.channels[input_channels[i]].first())
    }

    /**
     * Fire an operator in a given configuration
     */
    #[verifier(opaque)]
    pub open spec fn step(self, op: OperatorIndex) -> Configuration
        recommends
            self.valid(),
            self.fireable(op),
    {
        let input_channels = self.get_op_input_channels(op);
        let output_channels = self.get_op_output_channels(op);
        let inputs = self.get_op_input_values(op);

        let (outputs, operators, memory) = match self.operators[op] {
            Operator::NonMemory { state } => {
                let (next_state, outputs) = state_computation(op, state, inputs);
                (outputs, self.operators.insert(op, Operator::NonMemory { state: next_state }), self.memory)
            },

            Operator::Read { .. } => {
                let address_value = inputs.first();
                let output_value = self.memory[address_value.as_address()];
                (seq![output_value], self.operators, self.memory)
            },

            Operator::Write { .. } => {
                let address_value = inputs[0];
                let value = inputs[1];
                (seq![write_default_output()], self.operators, self.memory.insert(address_value.as_address(), value))
            },
        };

        // Pop input channels first
        let input_updated_channels = Map::new(|channel: ChannelIndex| self.is_channel(channel), |channel: ChannelIndex|
            if input_channels.contains(channel) { self.channels[channel].drop_first() }
            else { self.channels[channel] }
        );

        // Then push to output channels
        let output_updated_channels = Map::new(|channel: ChannelIndex| self.is_channel(channel), |channel: ChannelIndex|
            if output_channels.contains(channel) {
                let output_index = output_channels.index_of(channel);
                input_updated_channels[channel].push(outputs[output_index])
            }
            else { input_updated_channels[channel] }
        );

        Configuration {
            graph: self.graph,
            operators: operators,
            channels: output_updated_channels,
            memory: memory,
        }
    }

    /**
     * Lemma: Stepping a valid configuration gives a valid configuration.
     */
    pub proof fn lemma_step_valid(self, op: OperatorIndex)
        requires
            self.valid(),
            self.fireable(op),

        ensures
            self.graph =~~= self.step(op).graph,
            self.step(op).valid(),
    {
        reveal(Configuration::step);

        if channel_bound() > 0 {
            assert(forall |channel: ChannelIndex| self.is_channel(channel) ==>
                (#[trigger] self.step(op).channels[channel]).len() <= channel_bound())
        }
    }

    /**
     * Lemma: If two different operators op1 and op2 are fireable in a config
     * then firing either one of them would not change the input, output, state, and fireability
     * of the other.
     */
    pub proof fn lemma_step_independence(self, op1: OperatorIndex, op2: OperatorIndex)
        requires
            self.valid(),
            self.fireable(op1),
            self.fireable(op2),
            op1 != op2,

        ensures
            self.step(op1).fireable(op2),
            self.step(op2).fireable(op1),
            self.get_op_input_channels(op1) =~= self.step(op2).get_op_input_channels(op1),
            self.get_op_input_channels(op2) =~= self.step(op1).get_op_input_channels(op2),
            self.get_op_output_channels(op1) =~= self.step(op2).get_op_output_channels(op1),
            self.get_op_output_channels(op2) =~= self.step(op1).get_op_output_channels(op2),
            self.operators[op1] == self.step(op2).operators[op1],
            self.operators[op2] == self.step(op1).operators[op2],
            self.get_op_input_values(op1) =~= self.step(op2).get_op_input_values(op1),
            self.get_op_input_values(op2) =~= self.step(op1).get_op_input_values(op2),
    {
        reveal(Configuration::step);
        assert(self.operators[op1] == self.step(op2).operators[op1]);
        assert(self.operators[op2] == self.step(op1).operators[op2]);
        assert(self.step(op1).fireable(op2));
        assert(self.step(op2).fireable(op1));
        // assert(self.get_op_input_values(op1) == self.step(op2).get_op_input_values(op1));
        // assert(self.get_op_input_values(op2) == self.step(op1).get_op_input_values(op2));
    }


    /**
     * Lemma: For two different and non-memor operators op1 and op2,
     * if both of them are fireable in a configuration, then their execution
     * is commutable.
     *
     * i.e. without memory operators, dataflow graph execution is locally confluent.
     */
    pub proof fn lemma_step_non_memory_commute(self, op1: OperatorIndex, op2: OperatorIndex)
        requires
            self.valid(),
            self.fireable(op1),
            self.fireable(op2),
            op1 != op2,

            self.operators[op1].is_NonMemory() ||
            self.operators[op2].is_NonMemory() ||
            (self.operators[op1].is_Read() && self.operators[op2].is_Read()),

        ensures
            self.step(op1).step(op2) == self.step(op2).step(op1),
    {
        self.lemma_step_independence(op1, op2);

        assert(self.step(op1).step(op2) == self.step(op2).step(op1)) by {
            reveal(Configuration::step);
            assert(self.step(op1).step(op2).operators =~= self.step(op2).step(op1).operators);
            assert(self.step(op1).step(op2).channels =~~= self.step(op2).step(op1).channels);
            assert(self.step(op1).step(op2).memory =~= self.step(op2).step(op1).memory);
        }
    }
}

}
