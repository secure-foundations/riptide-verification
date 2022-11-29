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
    function FirePE(idx: PEIndex): (result: DataflowProgramState)
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

        ensures forall i :: 0 <= i < |result.channels| &&
                            i !in processingElements[idx].WaitingInputs() &&
                            i !in processingElements[idx].Outputs() ==> result.channels[i] == channels[i]
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

            case TrueSteerOperator(d, a, outputs) =>
                assert d in pe.Inputs() && a in pe.Inputs();
                assert outputs <= pe.Outputs();

                var valD := channels[d][0];
                var valA := channels[a][0];
                var newChannels := channels[d := channels[d][1..]][a := channels[a][1..]];
                var newState := DataflowProgramState(newChannels, processingElements);

                if valD == 1 then
                    newState.Multicast(valA , outputs)
                else
                    newState

            case FalseSteerOperator(d, a, outputs) =>
                assert d in pe.Inputs() && a in pe.Inputs();
                assert outputs <= pe.Outputs();

                var valD := channels[d][0];
                var valA := channels[a][0];
                var newChannels := channels[d := channels[d][1..]][a := channels[a][1..]];
                var newState := DataflowProgramState(newChannels, processingElements);

                if valD == 0 then
                    newState.Multicast(valA , outputs)
                else
                    newState
            
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
        ensures FirePE(idx1).IsFireable(idx2)
    {}

    // Output values do not depend on the channels other than WaitingInputs()
    lemma FirePEDependency(state1: DataflowProgramState, state2: DataflowProgramState, idx: PEIndex)
        requires state1.Wellformed() && state2.Wellformed()
        requires |state1.channels| == |state2.channels|
        requires 0 <= idx < |state1.processingElements| == |state2.processingElements|
        requires state1.processingElements[idx] == state2.processingElements[idx]
        requires state1.IsFireable(idx) && state2.IsFireable(idx)

        // Same input channels
        requires forall i :: i in state1.processingElements[idx].WaitingInputs() ==>
                             state1.channels[i] == state2.channels[i]

        // Same output channels
        requires forall i :: i in state1.processingElements[idx].Outputs() ==>
                             state1.channels[i] == state2.channels[i]

        // Would result in the same output input channels
        ensures var result1 := state1.FirePE(idx);
                var result2 := state2.FirePE(idx);
                forall i :: i in state1.processingElements[idx].WaitingInputs() ||
                            i in state1.processingElements[idx].Outputs() ==>
                            result1.channels[i] == result2.channels[i]

        // And the same final state for the fired PE
        ensures state1.FirePE(idx).processingElements[idx] == state2.FirePE(idx).processingElements[idx]
    {}

    // Output values do not depend on the channels other than WaitingInputs()
    lemma FirePEDependency2(state1: DataflowProgramState, state2: DataflowProgramState, idx: PEIndex)
        requires state1.Wellformed() && state2.Wellformed()
        requires |state1.channels| == |state2.channels|
        requires 0 <= idx < |state1.processingElements| == |state2.processingElements|
        requires state1.processingElements[idx] == state2.processingElements[idx]
        requires state1.IsFireable(idx) && state2.IsFireable(idx)

        // Same input values
        requires forall i :: i in state1.processingElements[idx].WaitingInputs() ==>
                             state1.channels[i][0] == state2.channels[i][0]

        // Same output channels
        // requires forall i :: i in state1.processingElements[idx].Outputs() ==>
        //                      state1.channels[i] == state2.channels[i]

        // Would result in the same "output values"
        ensures var pe := state1.processingElements[idx];
                var result1 := state1.FirePE(idx);
                var result2 := state2.FirePE(idx);
                forall i :: i in pe.Outputs() ==>
                    // If the output channel is not an input channel
                    (i !in pe.WaitingInputs() ==>
                        // Either unchanged
                        (result1.channels[i] == state1.channels[i] && result2.channels[i] == state2.channels[i]) ||
                        // Or appended with the same value
                        (result1.channels[i][|result1.channels[i]| - 1] == result2.channels[i][|result2.channels[i]| - 1] &&
                         result1.channels[i][..|result1.channels[i]| - 1] == state1.channels[i] &&
                         result2.channels[i][..|result2.channels[i]| - 1] == state2.channels[i])
                    ) ||
                    // Otherwise the output channel is an input channel
                    (i in pe.WaitingInputs() ==>
                        // Either has one value popped
                        (result1.channels[i] == state1.channels[i][1..] && result2.channels[i] == state2.channels[i][1..]) ||
                        // Or has one value popped and appended with the a value identical in result1 and result2
                        (result1.channels[i][|result1.channels[i]| - 1] == result2.channels[i][|result2.channels[i]| - 1] &&
                         result1.channels[i][..|result1.channels[i]| - 1] == state1.channels[i][1..] &&
                         result2.channels[i][..|result2.channels[i]| - 1] == state2.channels[i][1..])
                    )

        // And the same final state for the fired PE
        ensures state1.FirePE(idx).processingElements[idx] == state2.FirePE(idx).processingElements[idx]
    {}

    // This might need longer timeouts
    lemma FirePEConfluence(idx1: PEIndex, idx2: PEIndex)
        requires Wellformed()
        requires idx1 != idx2
        requires 0 <= idx1 < |processingElements|
        requires 0 <= idx2 < |processingElements|
        requires IsFireable(idx1)
        requires IsFireable(idx2)
        // ensures FirePE(idx1).FirePE(idx2) == FirePE(idx2).FirePE(idx1)
    {
        CommutableFiring(idx1, idx2);
        CommutableFiring(idx2, idx1);

        var fired1 := FirePE(idx1);
        var fired2 := FirePE(idx2);
        var fired12 := fired1.FirePE(idx2);
        var fired21 := fired2.FirePE(idx1);

        var inputs1 := processingElements[idx1].WaitingInputs();
        var inputs2 := processingElements[idx2].WaitingInputs();
        var outputs1 := processingElements[idx1].Outputs();
        var outputs2 := processingElements[idx2].Outputs();

        assert forall i :: i in inputs1 ==> i !in inputs2;
        assert forall i :: i in outputs1 ==> i !in outputs2;

        assert forall i :: 0 <= i < |channels| &&
                           i !in inputs1 &&
                           i !in inputs2 &&
                           i !in outputs1 &&
                           i !in outputs2 ==>
               fired12.channels[i] == fired21.channels[i] == channels[i];

        assume forall i :: i in inputs1 ==> i !in outputs2;
        assume forall i :: i in inputs2 ==> i !in outputs1;

        // assert forall i :: i in inputs1 ==> fired12.channels[i] == fired1.channels[i];
        // assert forall i :: i in outputs1 ==> fired12.channels[i] == fired1.channels[i];
        // assert forall i :: i in inputs2 ==> fired21.channels[i] == fired2.channels[i];
        // assert forall i :: i in outputs2 ==> fired21.channels[i] == fired2.channels[i];

        // assert forall i :: i in inputs1 ==> fired2.channels[i] == channels[i];
        // assert forall i :: i in outputs1 ==> fired2.channels[i] == channels[i];

        FirePEDependency2(fired2, this, idx1);
        FirePEDependency2(fired1, this, idx2);

        assert forall i :: i in inputs1 ==> fired21.channels[i] == fired1.channels[i];
        // assert forall i :: i in outputs1 ==> fired21.channels[i] == fired1.channels[i];

        // assert forall i :: 0 <= i < |channels| && i in outputs2 ==>
        //        fired21.channels[i] == fired2.channels[i];

        // match (processingElements[idx1], processingElements[idx2]) {
        //     case (AddOperator(_, _, _), AddOperator(_, _, _)) => {
                
        //     }
        //     // case (AddOperator(_, _, _), CarryOperator(_, _, _, _, _)) => {}
        //     // case (CarryOperator(_, _, _, _, _), AddOperator(_, _, _)) => {}
        //     // case (CarryOperator(_, _, _, _, _), CarryOperator(_, _, _, _, _)) => {}
        //     case _ => {}
        // }
    }
}
