type ChannelIndex = int
type PEIndex = int

type Value = int

datatype CarryState = CarryI | CarryB1 | CarryB2
datatype SteerState = SteerI | SteerA | SteerB
datatype MergeState = MergeI | MergeA | MergeB

datatype ProcessingElement =
    AddOperator   (                                     a: ChannelIndex, b: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    CarryOperator (             carryState: CarryState, d: ChannelIndex, a: ChannelIndex, b: ChannelIndex, outputs: seq<ChannelIndex>) |
    SteerOperator (steer: bool, steerState: SteerState, d: ChannelIndex, a: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    SelectOperator(                                     d: ChannelIndex, a: ChannelIndex, b: ChannelIndex, outputs: seq<ChannelIndex>) |
    OrderOperator (                                     a: ChannelIndex, b: ChannelIndex,                  outputs: seq<ChannelIndex>) |
    MergeOperator (             mergeState: MergeState, d: ChannelIndex, a: ChannelIndex, b: ChannelIndex, outputs: seq<ChannelIndex>)
{
    // Get all input channel indices of a PE
    function method Inputs(): seq<ChannelIndex> {
        match this
            case AddOperator(a, b, _) => [a, b]
            case CarryOperator(_, d, a, b, _) => [d, a, b]
            case SteerOperator(_, _, d, a, _) => [d, a]
            case SelectOperator(d, a, b, _) => [d, a, b]
            case OrderOperator(a, b, _) => [a, b]
            case MergeOperator(_, d, a, b, _) => [d, a, b]
    }

    // Get all output channel indices of a PE
    function method Outputs(): seq<ChannelIndex> {
        match this
            case AddOperator(_, _, outputs) => outputs
            case CarryOperator(_, _, _, _, outputs) => outputs
            case SteerOperator(_, _, _, _, outputs) => outputs
            case SelectOperator(_, _, _, outputs) => outputs
            case OrderOperator(_, _, output) => outputs
            case MergeOperator(_, _, _, _, outputs) => outputs
    }

    // Get the input channel indices a PE is currently waiting on
    function method WaitingInputs(): seq<ChannelIndex>
        ensures forall i :: i in WaitingInputs() ==> i in Inputs()
    {
        match this {
            case AddOperator(a, b, _) => [a, b]
            case CarryOperator(state, d, a, b, _) =>
                match state {
                    case CarryI => [a]
                    case CarryB1 => [d]
                    case CarryB2 => [b]
                }
            case SteerOperator(_, state, d, a, _) =>
                match state {
                    case SteerI => [d]
                    case SteerA => [a]
                    case SteerB => [a]
                }
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

    function method WaitingOutputs(): seq<ChannelIndex>
        ensures forall i :: i in WaitingOutputs() ==> i in Outputs()
    {
        match this {
            case AddOperator(_, _, outputs) => outputs
            case CarryOperator(state, _, _, _, outputs) =>
                match state {
                    case CarryI => outputs
                    case CarryB1 => []
                    case CarryB2 => outputs
                }
            case SteerOperator(_, state, _, _, outputs) =>
                match state {
                    case SteerI => []
                    case SteerA => outputs
                    case SteerB => []
                }
            case SelectOperator(_, _, _, outputs) => outputs
            case OrderOperator(_, _, outputs) => outputs
            case MergeOperator(state, _, _, _, outputs) =>
                match state {
                    case MergeI => []
                    case MergeA => outputs
                    case MergeB => outputs
                }
        }
    }
}

type Channel = seq<Value>

// datatype Channel = Channel(buffer: seq<Value>)
// {
//     function method Length(): int
//     {
//         |buffer|
//     }

//     function method Send(value: Value): (newChannel: Channel)
//     {
//         Channel(buffer + [value])
//     }

//     function method Receive(): (Value, Channel)
//         requires |buffer| > 0
//     {
//         (buffer[0], Channel(buffer[1..]))
//     }
// }

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
            forall pe :: pe in processingElements ==>
            forall idx :: idx in pe.Inputs() || idx in pe.Outputs() ==>
            0 <= idx < |channels|;

        var noIOOverlap := 
            forall pe1 :: pe1 in processingElements ==>
            forall pe2 :: pe2 in processingElements && pe1 != pe2 ==>
                (forall i :: i in pe1.Inputs() ==> i !in pe2.Inputs()) &&
                (forall i :: i in pe1.Outputs() ==> i !in pe2.Outputs());

        var distinctPEs :=
            forall i, j :: 0 <= i < j < |processingElements| ==>
            processingElements[i] != processingElements[j];

        var distinctInputChannels :=
            forall pe :: pe in processingElements ==>
            forall i, j :: 0 <= i < j < |pe.Inputs()| ==>
            pe.Inputs()[i] != pe.Inputs()[j];

        var distinctOutputChannels :=
            forall pe :: pe in processingElements ==>
            forall i, j :: 0 <= i < j < |pe.Outputs()| ==>
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
            |channels[ChannelIndex]| != 0
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
                    (idx in outputs ==> result.channels[idx] == channels[idx] + [value])

        decreases outputs
    {
        if |outputs| == 0 then
            this
        else
            var idx := outputs[0];
            assert idx in outputs;
            var newChannels := channels[idx := channels[idx] + [value]];
            var newState := DataflowProgramState(newChannels, processingElements);
            assert forall idx :: idx in outputs[1..] ==> idx in outputs;
            newState.Multicast(value, outputs[1..])
    }

    // Fires the specified PE and transition to a new state
    function {:opaque} FirePE(idx: PEIndex): (result: DataflowProgramState)
        requires 0 <= idx < |processingElements|
        requires Wellformed()
        requires IsFireable(idx)

        ensures Wellformed()
        ensures |result.channels| == |channels|
        ensures |result.processingElements| == |processingElements|

        ensures result.processingElements[idx].Inputs() == processingElements[idx].Inputs()
        ensures result.processingElements[idx].Outputs() == processingElements[idx].Outputs()

        ensures forall i :: 0 <= i < |result.processingElements| && i != idx
                            ==> result.processingElements[i] == processingElements[i]

        // Characterize how the channels change after firing
        ensures forall i :: i in processingElements[idx].WaitingInputs() && i in processingElements[idx].WaitingOutputs() ==>
                            |result.channels[i]| >= 1 && result.channels[i][..|result.channels[i]| - 1] == channels[i][1..]

        ensures forall i :: i in processingElements[idx].WaitingInputs() && i !in processingElements[idx].WaitingOutputs() ==>
                            result.channels[i] == channels[i][1..]

        ensures forall i :: i !in processingElements[idx].WaitingInputs() && i in processingElements[idx].WaitingOutputs() ==>
                            |result.channels[i]| >= 1 && result.channels[i][..|result.channels[i]| - 1] == channels[i]

        ensures forall i :: i !in processingElements[idx].WaitingInputs() && i !in processingElements[idx].WaitingOutputs() && 0 <= i < |result.channels| ==>
                            result.channels[i] == channels[i]
    {
        var pe := processingElements[idx];
        match pe
            case AddOperator(a, b, outputs) =>
                assert a in pe.Inputs() && b in pe.Inputs();
                assert outputs <= pe.Outputs();
                // assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
                // assert a != b;

                var valA := channels[a][0];
                var valB := channels[b][0];
                var newChannels := channels[a := channels[a][1..]][b := channels[b][1..]];
                var newState := DataflowProgramState(newChannels, processingElements);

                newState.Multicast(valA + valB, outputs)

            case SteerOperator(steer, state, d, a, outputs) =>
                match state {
                    case SteerI =>
                        assert d in pe.Inputs();

                        var valD := channels[d][0];
                        var newChannels := channels[d := channels[d][1..]];
                        var updatedPE := if (valD == 0 && !steer) || (valD != 0 && steer)
                                            then SteerOperator(steer, SteerA, d, a, outputs)
                                            else SteerOperator(steer, SteerB, d, a, outputs);
                        var newPEs := processingElements[idx := updatedPE];
                        
                        DataflowProgramState(newChannels, newPEs)

                    case SteerA =>
                        assert a in pe.Inputs();
                        var valA := channels[a][0];
                        var newChannels := channels[a := channels[a][1..]];
                        var newPEs := processingElements[idx := SteerOperator(steer, SteerI, d, a, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);
                        
                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        assert processingElements[idx].Outputs() == outputs;

                        newState.Multicast(valA , outputs)

                    case SteerB =>
                        assert a in pe.Inputs();
                        var newChannels := channels[a := channels[a][1..]];
                        var newPEs := processingElements[idx := SteerOperator(steer, SteerI, d, a, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);
                        newState

                }

            case SelectOperator(d, a, b, outputs) =>
                assert d in pe.Inputs() && a in pe.Inputs() && b in pe.Inputs();
                assert outputs <= pe.Outputs();

                var valD := channels[d][0];
                var valA := channels[a][0];
                var valB := channels[b][0];
                var newChannels := channels[d := channels[d][1..]][a := channels[a][1..]][b := channels[b][1..]];
                var newState := DataflowProgramState(newChannels, processingElements);

                if valD == 1 then
                    newState.Multicast(valA, outputs)
                else
                    newState.Multicast(valB, outputs)

            case CarryOperator(state, d, a, b, outputs) =>
                match state {
                    case CarryI =>
                        assert a in pe.Inputs();

                        var valA := channels[a][0];
                        var newChannels := channels[a := channels[a][1..]];
                        var newPEs := processingElements[idx := CarryOperator(CarryB1, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        assert processingElements[idx].Outputs() == outputs;

                        newState.Multicast(valA, outputs)

                    case CarryB1 =>
                        assert d in pe.Inputs();

                        var valD := channels[d][0];
                        var newChannels := channels[d := channels[d][1..]];

                        var updatedPE := if valD == 0 then CarryOperator(CarryI, d, a, b, outputs)
                                                      else CarryOperator(CarryB2, d, a, b, outputs);
                        var newPEs := processingElements[idx := updatedPE];

                        DataflowProgramState(newChannels, newPEs)
                        
                    case CarryB2 =>
                        assert b in pe.Inputs();

                        var valB := channels[b][0];
                        var newChannels := channels[b := channels[b][1..]];
                        var newPEs := processingElements[idx := CarryOperator(CarryB1, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        assert processingElements[idx].Outputs() == outputs;

                        newState.Multicast(valB, outputs)
                }

            case OrderOperator(a, b, outputs) =>
                assert a in pe.Inputs() && b in pe.Inputs();
                assert outputs <= pe.Outputs();
                
                var valA := channels[a][0];
                var valB := channels[b][0];
                var newChannels := channels[a := channels[a][1..]][b := channels[b][1..]];
                var newState := DataflowProgramState(newChannels, processingElements);

                newState.Multicast(valB, outputs)

            case MergeOperator(state, d, a, b, outputs) =>
                match state {
                    case MergeI =>
                        assert d in pe.Inputs();

                        var valD := channels[d][0];
                        var newChannels := channels[d := channels[d][1..]];

                        var updatedPE := if valD == 0 then MergeOperator(MergeA, d, a, b, outputs)
                                                      else MergeOperator(MergeB, d, a, b, outputs);
                        var newPEs := processingElements[idx := updatedPE];

                        DataflowProgramState(newChannels, newPEs)

                    case MergeA =>
                        assert a in pe.Inputs();

                        var valA := channels[a][0];
                        var newChannels := channels[a := channels[a][1..]];
                        var newPEs := processingElements[idx := MergeOperator(MergeI, d, a, b, outputs)];
                        var newState := DataflowProgramState(newChannels, newPEs);

                        assert forall i :: 0 <= i < |newPEs| ==> newPEs[i].Inputs() == processingElements[i].Inputs();
                        assert processingElements[idx].Outputs() == outputs;

                        newState.Multicast(valA, outputs)

                    case MergeB =>
                        assert b in pe.Inputs();
                        assert outputs <= pe.Outputs();

                        var valB := channels[b][0];
                        var newChannels := channels[b := channels[b][1..]];
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
        requires idx1 != idx2
        requires 0 <= idx1 < |processingElements|
        requires 0 <= idx2 < |processingElements|
        requires IsFireable(idx1)
        requires IsFireable(idx2)
        ensures FirePE(idx1).Wellformed()
        ensures FirePE(idx1).IsFireable(idx2)
    {
        reveal_FirePE();
    }

    // Output values do not depend on the channels other than WaitingInputs()
    lemma FirePEDependency(state1: DataflowProgramState, state2: DataflowProgramState, idx: PEIndex)
        requires state1.Wellformed() && state2.Wellformed()
        requires |state1.channels| == |state2.channels|
        requires 0 <= idx < |state1.processingElements| == |state2.processingElements|
        requires state1.processingElements[idx] == state2.processingElements[idx]
        requires state1.IsFireable(idx) && state2.IsFireable(idx)

        // Same input values
        requires forall i :: i in state1.processingElements[idx].WaitingInputs() ==>
                             state1.channels[i][0] == state2.channels[i][0]

        // Would result in the same output values and final states
        ensures var pe := state1.processingElements[idx];
                var result1 := state1.FirePE(idx);
                var result2 := state2.FirePE(idx);
                forall i :: i in pe.WaitingOutputs() ==>
                result1.channels[i][|result1.channels[i]| - 1] == result2.channels[i][|result2.channels[i]| - 1] &&
                result1.processingElements[idx] == result2.processingElements[idx]
    {
        reveal_FirePE();
    }

    // This might need longer timeouts
    lemma FirePEConfluence(idx1: PEIndex, idx2: PEIndex)
        requires Wellformed()
        requires idx1 != idx2
        requires 0 <= idx1 < |processingElements|
        requires 0 <= idx2 < |processingElements|
        requires IsFireable(idx1)
        requires IsFireable(idx2)

        ensures FirePE(idx1).Wellformed()
        ensures FirePE(idx2).Wellformed()
        ensures FirePE(idx1).IsFireable(idx2)
        ensures FirePE(idx2).IsFireable(idx1)
        ensures FirePE(idx1).FirePE(idx2) == FirePE(idx2).FirePE(idx1)
    {
        CommutableFiring(idx1, idx2);
        CommutableFiring(idx2, idx1);
        
        var fired12 := FirePE(idx1).FirePE(idx2);
        var fired21 := FirePE(idx2).FirePE(idx1);

        assert fired12 == fired21 by {
            var pe1 := processingElements[idx1];
            var pe2 := processingElements[idx2];

            var inputs1 := pe1.WaitingInputs();
            var inputs2 := pe2.WaitingInputs();
            var outputs1 := pe1.WaitingOutputs();
            var outputs2 := pe2.WaitingOutputs();

            forall i | 0 <= i < |fired12.channels|
                ensures fired12.channels[i] == fired21.channels[i]
            {
                reveal_FirePE();
                if i in inputs1 && i in outputs1 {}
                else if i in inputs2 && i in outputs2 {}
                else if i in inputs1 && i in outputs2 {}
                else if i in inputs2 && i in outputs1 {}
                else {}
            }

            assert fired12.channels == fired21.channels;
            assert fired12.processingElements == fired21.processingElements by {
                reveal_FirePE();
            }
        }

        // assert FirePE(idx1).Wellformed();
        // assert FirePE(idx2).Wellformed();
        // assert FirePE(idx1).IsFireable(idx2);
        // assert FirePE(idx2).IsFireable(idx1);
        // assert FirePE(idx1).FirePE(idx2) == FirePE(idx2).FirePE(idx1);
    }
}
