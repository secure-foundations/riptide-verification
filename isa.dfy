type ChannelIndex = int
type PEIndex = int

type Value = int

datatype CarryState = CarryI | CarryB1 | CarryB2
datatype MergeState = MergeI | MergeA | MergeB

datatype ProcessingElement =
    AddOperator       (                        a: ChannelIndex, b: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    CarryOperator     (carryState: CarryState, d: ChannelIndex, a: ChannelIndex, b: ChannelIndex, outputs: seq<ChannelIndex>) |
    TrueSteerOperator (                        d: ChannelIndex, a: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    FalseSteerOperator(                        d: ChannelIndex, a: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    SelectOperator    (                        d: ChannelIndex, a: ChannelIndex, b: ChannelIndex, outputs: seq<ChannelIndex>) |
    OrderOperator     (                        a: ChannelIndex, b: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    MergeOperator     (mergeState: MergeState, d: ChannelIndex, a: ChannelIndex, b: ChannelIndex, outputs: seq<ChannelIndex>)
{
    // Get all input channel indices of a PE
    function method Inputs(): seq<ChannelIndex> {
        match this
            case AddOperator(a, b, _) => [a, b]
            case CarryOperator(_, d, a, b, _) => [d, a, b]
            case TrueSteerOperator(d, a, _) => [d, a]
            case FalseSteerOperator(d, a, _) => [d, a]
            case SelectOperator(d, a, b, _) => [d, a, b]
            case OrderOperator(a, b, _) => [a, b]
            case MergeOperator(_, d, a, b, _) => [d, a, b]
    }

    // Get all output channel indices of a PE
    function method Outputs(): seq<ChannelIndex> {
        match this
            case AddOperator(_, _, outputs) => outputs
            case CarryOperator(_, _, _, _, outputs) => outputs
            case TrueSteerOperator(_, _, outputs) => outputs
            case FalseSteerOperator(_, _, outputs) => outputs
            case SelectOperator(_, _, _, outputs) => outputs
            case OrderOperator(_, _, output) => outputs
            case MergeOperator(_, _, _, _, outputs) => outputs
    }

    // Get the input channel indices a PE is currently waiting on
    function method WaitingInputs(): seq<ChannelIndex>
        ensures multiset(WaitingInputs()) <= multiset(Inputs())
    {
        match this {
            case AddOperator(a, b, _) => [a, b]
            case CarryOperator(state, d, a, b, _) =>
                match state {
                    case CarryI => [a]
                    case CarryB1 => [d]
                    case CarryB2 => [b]
                }
            case TrueSteerOperator(d, a, _) => [d, a]
            case FalseSteerOperator(d, a, _) => [d, a]
            case SelectOperator(d, a, b, _) => [d, a, b]
            case OrderOperator(a, b, _) => [a, b]
            case MergeOperator(state, d, a, b, _) =>
                match state {
                    case MergeI => [d]
                    case MergeA => [a]
                    case MergeB => [b]
                }
        }
    }
}

datatype Channel = Channel(buffer: seq<Value>)
{
    function method Length(): int
    {
        |buffer|
    }

    function method Send(value: Value): (newChannel: Channel)
    {
        Channel(buffer + [value])
    }

    function method Receive(): (Value, Channel)
        requires |buffer| > 0
    {
        (buffer[0], Channel(buffer[1..]))
    }
}

datatype DataflowProgramState = DataflowProgramState(channels: seq<Channel>, processingElements: seq<ProcessingElement>)
{
    // Returns if the dataflow program is well-formed
    predicate Wellformed()
    {
        // Inputs & outputs are valid channel indices
        // Inputs do not overlap
        // Outputs do not overlap
        // Each node has distinct input channels

        var validChannelIndices :=
            forall pe :: pe in processingElements[..] ==>
            forall idx :: idx in pe.Inputs() || idx in pe.Outputs() ==>
            0 <= idx < |channels|;

        var noIOOverlap := 
            forall pe1 :: pe1 in processingElements[..] ==>
            forall pe2 :: pe2 in processingElements[..] && pe1 != pe2 ==>
                multiset(pe1.Inputs()) * multiset(pe2.Inputs()) == multiset{} &&
                multiset(pe1.Outputs()) * multiset(pe2.Outputs()) == multiset{};

        var distinctPEs :=
            forall i, j :: 0 <= i < j < |processingElements| ==>
            processingElements[i] != processingElements[j];

        var distinctInputChannels :=
            forall pe :: pe in processingElements[..] ==>
            forall i, j :: 0 <= i < j < |pe.Inputs()| ==>
            pe.Inputs()[i] != pe.Inputs()[j];

        var distinctOutputChannels :=
            forall pe :: pe in processingElements[..] ==>
            forall i, j :: 0 <= i < |pe.Outputs()| && 0 <= j < |pe.Outputs()| ==>
            pe.Outputs()[i] != pe.Outputs()[j];

        validChannelIndices && noIOOverlap && distinctPEs && distinctInputChannels && distinctOutputChannels
    }

    // Returns if the given PE is fireable in the dataflow program
    predicate IsFireable(idx: PEIndex)
        requires Wellformed()
        requires 0 <= idx < |processingElements|
    {
        forall ChannelIndex :: ChannelIndex in processingElements[idx].WaitingInputs() ==>
            ChannelIndex in processingElements[idx].Inputs() && // TODO: this is not necessary
            channels[ChannelIndex].Length() != 0
    }

    function Multicast(value: Value, outputs: seq<ChannelIndex>): DataflowProgramState
        requires Wellformed()
        requires forall idx :: idx in outputs ==> 0 <= idx < |channels|
        requires forall i, j :: 0 <= i < j < |outputs| ==> outputs[i] != outputs[j]

        ensures var result := Multicast(value, outputs);
                |result.channels| == |channels| &&
                result.processingElements == processingElements &&
                forall idx :: 0 <= idx < |channels| ==>
                    (idx !in outputs ==> result.channels[idx] == channels[idx]) &&
                    (idx in outputs ==> result.channels[idx] == channels[idx].Send(value))

        decreases outputs
    {
        if |outputs| == 0 then
            this
        else
            var idx := outputs[0];
            assert idx in outputs;
            var newChannels := channels[idx := channels[idx].Send(value)];
            var newState := DataflowProgramState(newChannels, processingElements);
            assert forall idx :: idx in outputs[1..] ==> idx in outputs;
            newState.Multicast(value, outputs[1..])
    }

    // Fires the specified PE and transition to a new state
    function FirePE(idx: PEIndex): (result: DataflowProgramState)
        requires 0 <= idx < |processingElements|
        requires Wellformed()
        requires IsFireable(idx)
        ensures Wellformed()
        ensures |result.channels| == |channels|
        ensures |result.processingElements| == |processingElements|

        ensures result.processingElements[idx].Inputs() == processingElements[idx].Inputs()
        ensures result.processingElements[idx].Outputs() == processingElements[idx].Outputs()

        ensures forall i :: 0 <= i < |result.channels| &&
                            i !in processingElements[idx].Inputs() &&
                            i !in processingElements[idx].Outputs() ==> result.channels[i] == channels[i]
    {
        var pe := processingElements[idx];
        match pe
            case AddOperator(a, b, outputs) =>
                assert a in pe.Inputs() && b in pe.Inputs();
                assert outputs <= pe.Outputs();
                // assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
                // assert a != b;

                var (valA, newChannelA) := channels[a].Receive();
                var (valB, newChannelB) := channels[b].Receive();
                var newChannels := channels[a := newChannelA][b := newChannelB];
                var newState := DataflowProgramState(newChannels, processingElements);

                newState.Multicast(valA + valB, outputs)

            case TrueSteerOperator(d, a, outputs) =>
                assert d in pe.Inputs() && a in pe.Inputs();
                assert outputs <= pe.Outputs();

                var (valD, newChannelD) := channels[d].Receive();
                var (valA, newChannelA) := channels[a].Receive();
                var newChannels := channels[d := newChannelD][a := newChannelA];
                var newState := DataflowProgramState(newChannels, processingElements);

                if valD == 1 then
                    newState.Multicast(valA , outputs)
                else
                    newState

            case FalseSteerOperator(d, a, outputs) =>
                assert d in pe.Inputs() && a in pe.Inputs();
                assert outputs <= pe.Outputs();

                var (valD, newChannelD) := channels[d].Receive();
                var (valA, newChannelA) := channels[a].Receive();
                var newChannels := channels[d := newChannelD][a := newChannelA];
                var newState := DataflowProgramState(newChannels, processingElements);

                if valD == 0 then
                    newState.Multicast(valA , outputs)
                else
                    newState
            
            case SelectOperator(d, a, b, outputs) =>
                assert d in pe.Inputs() && a in pe.Inputs() && b in pe.Inputs();
                assert outputs <= pe.Outputs();

                var (valD, newChannelD) := channels[d].Receive();
                var (valA, newChannelA) := channels[a].Receive();
                var (valB, newChannelB) := channels[b].Receive();
                var newChannels := channels[d := newChannelD][a := newChannelA][b := newChannelB];
                var newState := DataflowProgramState(newChannels, processingElements);

                if valD == 1 then
                    newState.Multicast(valA, outputs)
                else
                    newState.Multicast(valB, outputs)

            case CarryOperator(state, d, a, b, outputs) =>
                match state {
                    case CarryI =>
                        assert a in pe.Inputs();

                        var (valA, newChannelA) := channels[a].Receive();
                        var newChannels := channels[a := newChannelA];
                        var newPEs := processingElements[idx := CarryOperator(CarryB1, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        newState.Multicast(valA, outputs)

                    case CarryB1 =>
                        assert d in pe.Inputs();

                        var (valD, newChannelD) := channels[d].Receive();
                        var newChannels := channels[d := newChannelD];

                        var updatedPE := if valD == 0 then CarryOperator(CarryI, d, a, b, outputs)
                                                      else CarryOperator(CarryB2, d, a, b, outputs);
                        var newPEs := processingElements[idx := updatedPE];

                        DataflowProgramState(newChannels, newPEs)
                        
                    case CarryB2 =>
                        assert b in pe.Inputs();

                        var (valB, newChannelB) := channels[b].Receive();
                        var newChannels := channels[b := newChannelB];
                        var newPEs := processingElements[idx := CarryOperator(CarryB1, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        newState.Multicast(valB, outputs)
                }

            case OrderOperator(a, b, outputs) =>
                assert a in pe.Inputs() && b in pe.Inputs();
                assert outputs <= pe.Outputs();
                
                var (valA, newChannelA) := channels[a].Receive();
                var (valB, newChannelB) := channels[b].Receive();
                var newChannels := channels[a := newChannelA][b := newChannelB];
                var newState := DataflowProgramState(newChannels, processingElements);

                newState.Multicast(valB, outputs)

            case MergeOperator(state, d, a, b, outputs) =>
                match state {
                    case MergeI =>
                        assert d in pe.Inputs();

                        var (valD, newChannelD) := channels[d].Receive();
                        var newChannels := channels[d := newChannelD];

                        var updatedPE := if valD == 0 then MergeOperator(MergeA, d, a, b, outputs)
                                                      else MergeOperator(MergeB, d, a, b, outputs);
                        var newPEs := processingElements[idx := updatedPE];

                        DataflowProgramState(newChannels, newPEs)

                    case MergeA =>
                        assert a in pe.Inputs();

                        var (valA, newChannelA) := channels[a].Receive();
                        var newChannels := channels[a := newChannelA];
                        var newPEs := processingElements[idx := MergeOperator(MergeI, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        newState.Multicast(valA, outputs)

                    case MergeB =>
                        assert b in pe.Inputs();

                        var (valB, newChannelB) := channels[b].Receive();
                        var newChannels := channels[b := newChannelB];
                        var newPEs := processingElements[idx := MergeOperator(MergeI, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        newState.Multicast(valB, outputs)
                }

    }

    predicate AreBothFireable(idx1: PEIndex, idx2: PEIndex)
        requires Wellformed()
    {
        0 <= idx1 < |processingElements| &&
        0 <= idx2 < |processingElements| &&
        IsFireable(idx1) && IsFireable(idx2)
    }

    lemma CommutableFiring(idx1: PEIndex, idx2: PEIndex)
        requires Wellformed()
        requires AreBothFireable(idx1, idx2)
        requires idx1 != idx2
        ensures FirePE(idx1).IsFireable(idx2)
    {}

    // This might need longer timeouts
    lemma FirePEConfluence(idx1: PEIndex, idx2: PEIndex)
        requires Wellformed()
        requires AreBothFireable(idx1, idx2)
        requires idx1 != idx2
        ensures FirePE(idx1).FirePE(idx2) == FirePE(idx2).FirePE(idx1)
    {
        CommutableFiring(idx1, idx2);
    }
}
