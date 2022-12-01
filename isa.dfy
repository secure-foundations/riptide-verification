type ChannelIndex = int
type PEIndex = int
type Value = int
type Channel = seq<Value>

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

datatype DataflowProgramState = DataflowProgramState(channels: seq<Channel>, processingElements: seq<ProcessingElement>)
{
    // Returns if the dataflow program is well-formed
    predicate {:opaque} WellFormed()
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
            forall i, j :: 0 <= i < j < |processingElements| ==>
                (forall k :: k in processingElements[i].Inputs() ==> k !in processingElements[j].Inputs()) &&
                (forall k :: k in processingElements[i].Outputs() ==> k !in processingElements[j].Outputs());

        var distinctInputChannels :=
            forall pe :: pe in processingElements ==>
            forall i, j :: 0 <= i < j < |pe.Inputs()| ==>
            pe.Inputs()[i] != pe.Inputs()[j];

        var distinctOutputChannels :=
            forall pe :: pe in processingElements ==>
            forall i, j :: 0 <= i < j < |pe.Outputs()| ==>
            pe.Outputs()[i] != pe.Outputs()[j];

        validChannelIndices && noIOOverlap && distinctInputChannels && distinctOutputChannels
    }

    // Returns if the given PE is fireable in the dataflow program
    predicate IsFireable(idx: PEIndex)
        requires WellFormed()
        requires 0 <= idx < |processingElements|
    {
        reveal_WellFormed();
        forall i :: i in processingElements[idx].WaitingInputs() ==> |channels[i]| != 0
    }

    function UpdatePE(idx: PEIndex, pe: ProcessingElement): (result: DataflowProgramState)
        requires WellFormed()
        requires 0 <= idx < |processingElements|
        requires pe.Inputs() == processingElements[idx].Inputs()
        requires pe.Outputs() == processingElements[idx].Outputs()
        ensures result.WellFormed()
    {
        reveal_WellFormed();
        DataflowProgramState(channels, processingElements[idx := pe])
    }

    function Receive(channelIdx: ChannelIndex): (Value, DataflowProgramState)
        requires WellFormed()
        requires 0 <= channelIdx < |channels|
        requires |channels[channelIdx]| > 0
        ensures Receive(channelIdx).1.WellFormed()
    {
        reveal_WellFormed();
        (channels[channelIdx][0], DataflowProgramState(channels[channelIdx := channels[channelIdx][1..]], processingElements))
    }

    function Send(value: Value, channelIdx: ChannelIndex): (result: DataflowProgramState)
        requires WellFormed()
        requires 0 <= channelIdx < |channels|
        ensures result.WellFormed()
    {
        reveal_WellFormed();
        DataflowProgramState(channels[channelIdx := channels[channelIdx] + [value]], processingElements)
    }

    function Multicast(value: Value, outputs: seq<ChannelIndex>): (result: DataflowProgramState)
        requires WellFormed()
        requires forall idx :: idx in outputs ==> 0 <= idx < |channels|
        requires forall i, j :: 0 <= i < j < |outputs| ==> outputs[i] != outputs[j]

        ensures |result.channels| == |channels| &&
                result.processingElements == processingElements &&
                forall idx :: 0 <= idx < |channels| ==>
                    (idx !in outputs ==> result.channels[idx] == channels[idx]) &&
                    (idx in outputs ==> result.channels[idx] == channels[idx] + [value])
        ensures result.WellFormed()

        decreases outputs
    {
        if |outputs| == 0 then
            this
        else
            var idx := outputs[0];
            assert idx in outputs;
            var newState := this.Send(value, idx);
            assert forall idx :: idx in outputs[1..] ==> idx in outputs;
            reveal_WellFormed();
            newState.Multicast(value, outputs[1..])
    }

    // Fires the specified PE and transition to a new state
    function {:opaque} FirePE(idx: PEIndex): (result: DataflowProgramState)
        requires 0 <= idx < |processingElements|
        requires WellFormed()
        requires IsFireable(idx)

        ensures |result.channels| == |channels|
        ensures |result.processingElements| == |processingElements|

        ensures result.processingElements[idx].Inputs() == processingElements[idx].Inputs()
        ensures result.processingElements[idx].Outputs() == processingElements[idx].Outputs()

        ensures forall i :: 0 <= i < |result.processingElements| && i != idx ==> result.processingElements[i] == processingElements[i]

        ensures result.WellFormed()
    {
        reveal_WellFormed();
        var pe := processingElements[idx];
        match pe
            case AddOperator(a, b, outputs) =>
                assert a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs == pe.Outputs();
                assert a == pe.Inputs()[0] && b == pe.Inputs()[1];

                var (valA, newState) := this.Receive(a);
                var (valB, newState) := newState.Receive(b);
                newState.Multicast(valA + valB, outputs)

            case SteerOperator(steer, state, d, a, outputs) =>
                match state {
                    case SteerI =>
                        assert d in pe.WaitingInputs();

                        var (valD, newState) := this.Receive(d);
                        var updatedPE := if (valD == 0 && !steer) || (valD != 0 && steer)
                                         then SteerOperator(steer, SteerA, d, a, outputs)
                                         else SteerOperator(steer, SteerB, d, a, outputs);
                        newState.UpdatePE(idx, updatedPE)

                    case SteerA =>
                        assert a in pe.Inputs();

                        var (valA, newState) := this.Receive(a);
                        var newState := newState.UpdatePE(idx, SteerOperator(steer, SteerI, d, a, outputs));
                        newState.Multicast(valA , outputs)

                    case SteerB =>
                        assert a in pe.Inputs();
                        var (_, newState) := this.Receive(a);
                        newState.UpdatePE(idx, SteerOperator(steer, SteerI, d, a, outputs))
                }

            case SelectOperator(d, a, b, outputs) =>
                assert d in pe.WaitingInputs() && a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs == pe.Outputs();
                assert d == pe.Inputs()[0] && a == pe.Inputs()[1] && b == pe.Inputs()[2];
                
                var (valD, newState) := this.Receive(d);
                var (valA, newState) := newState.Receive(a);
                var (valB, newState) := newState.Receive(b);

                if valD == 1 then
                    newState.Multicast(valA, outputs)
                else
                    newState.Multicast(valB, outputs)

            case CarryOperator(state, d, a, b, outputs) =>
                match state {
                    case CarryI =>
                        assert a in pe.WaitingInputs();

                        var (valA, newState) := this.Receive(a);
                        var newState := newState.UpdatePE(idx, CarryOperator(CarryB1, d, a, b, outputs));
                        newState.Multicast(valA, outputs)

                    case CarryB1 =>
                        assert d in pe.WaitingInputs();

                        var (valD, newState) := this.Receive(d);
                        var updatedPE := if valD == 0 then CarryOperator(CarryI, d, a, b, outputs)
                                                      else CarryOperator(CarryB2, d, a, b, outputs);
                        newState.UpdatePE(idx, updatedPE)
                        
                    case CarryB2 =>
                        assert b in pe.WaitingInputs();

                        var (valB, newState) := this.Receive(b);
                        var newState := newState.UpdatePE(idx, CarryOperator(CarryB1, d, a, b, outputs));
                        newState.Multicast(valB, outputs)
                }

            case OrderOperator(a, b, outputs) =>
                assert a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs <= pe.Outputs();
                assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
                
                var (valA, newState) := this.Receive(a);
                var (valB, newState) := newState.Receive(b);
                newState.Multicast(valB, outputs)

            case MergeOperator(state, d, a, b, outputs) =>
                match state {
                    case MergeI =>
                        assert d in pe.WaitingInputs();

                        var (valD, newState) := this.Receive(d);
                        var updatedPE := if valD == 0 then MergeOperator(MergeA, d, a, b, outputs)
                                                      else MergeOperator(MergeB, d, a, b, outputs);
                        newState.UpdatePE(idx, updatedPE)

                    case MergeA =>
                        assert a in pe.WaitingInputs();

                        var (valA, newState) := this.Receive(a);
                        var newState := newState.UpdatePE(idx, MergeOperator(MergeI, d, a, b, outputs));
                        newState.Multicast(valA, outputs)

                    case MergeB =>
                        assert b in pe.WaitingInputs();

                        var (valB, newState) := this.Receive(b);
                        var newState := newState.UpdatePE(idx, MergeOperator(MergeI, d, a, b, outputs));
                        newState.Multicast(valB, outputs)
                }
    }

    // Specifies how a particular channel changes after Fire(idx)
    lemma FirePEChannelChange(idx: PEIndex, channelIdx: ChannelIndex)
        requires 0 <= idx < |processingElements|
        requires 0 <= channelIdx < |channels|
        requires WellFormed()
        requires IsFireable(idx)

        ensures |FirePE(idx).channels| == |channels|
        ensures |FirePE(idx).processingElements| == |processingElements|

        ensures FirePE(idx).processingElements[idx].Inputs() == processingElements[idx].Inputs()
        ensures FirePE(idx).processingElements[idx].Outputs() == processingElements[idx].Outputs()

        ensures forall i :: 0 <= i < |FirePE(idx).processingElements| && i != idx
                        ==> FirePE(idx).processingElements[i] == processingElements[i]

        requires WellFormed()

        ensures var original := channels[channelIdx];
                var result := FirePE(idx).channels[channelIdx];
                var pe := processingElements[idx];
                
                (channelIdx in pe.WaitingInputs() && channelIdx in pe.WaitingOutputs() ==>
                 |result| >= 1 && result[..|result| - 1] == original[1..]) &&

                (channelIdx in pe.WaitingInputs() && channelIdx !in pe.WaitingOutputs() ==>
                 result == original[1..]) &&

                (channelIdx !in pe.WaitingInputs() && channelIdx in pe.WaitingOutputs() ==>
                 |result| >= 1 && result[..|result| - 1] == original) &&

                (channelIdx !in pe.WaitingInputs() && channelIdx !in pe.WaitingOutputs() ==>
                 result == original)
    {
        reveal_FirePE();

        var pe := processingElements[idx];
        match pe
            case AddOperator(a, b, outputs) => {
                assert a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs == pe.Outputs();
                assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
            }

            case SelectOperator(d, a, b, outputs) => {
                assert outputs == pe.Outputs();
                assert d == pe.Inputs()[0] && a == pe.Inputs()[1] && b == pe.Inputs()[2];
            }

            case OrderOperator(a, b, outputs) => {
                assert outputs <= pe.Outputs();
                assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
            }

            case _ => {}
    }

    lemma CommutableFiring(idx1: PEIndex, idx2: PEIndex)
        requires WellFormed()
        requires idx1 != idx2
        requires 0 <= idx1 < |processingElements|
        requires 0 <= idx2 < |processingElements|
        requires IsFireable(idx1)
        requires IsFireable(idx2)
        ensures FirePE(idx1).WellFormed()
        ensures FirePE(idx1).IsFireable(idx2)
    {
        reveal_FirePE();
    }

    // Output values do not depend on the channels other than WaitingInputs()
    lemma FirePEDependency(state1: DataflowProgramState, state2: DataflowProgramState, idx: PEIndex, outputIdx: ChannelIndex)
        requires state1.WellFormed() && state2.WellFormed()
        requires |state1.channels| == |state2.channels|
        requires 0 <= idx < |state1.processingElements| == |state2.processingElements|
        requires state1.processingElements[idx] == state2.processingElements[idx]
        requires state1.IsFireable(idx) && state2.IsFireable(idx)
        requires outputIdx in state1.processingElements[idx].WaitingOutputs()

        // Same input values
        requires forall i :: i in state1.processingElements[idx].WaitingInputs() ==>
                             state1.channels[i][0] == state2.channels[i][0]

        // Would result in the same output values and final states
        ensures var pe := state1.processingElements[idx];
                var result1 := state1.FirePE(idx);
                var result2 := state2.FirePE(idx);
                result1.WellFormed() &&
                result2.WellFormed() &&
                |result1.channels[outputIdx]| > 0 &&
                |result2.channels[outputIdx]| > 0 &&
                result1.channels[outputIdx][|result1.channels[outputIdx]| - 1] == result2.channels[outputIdx][|result2.channels[outputIdx]| - 1] &&
                result1.processingElements[idx] == result2.processingElements[idx]
    {
        reveal_FirePE();

        var pe := state1.processingElements[idx];
        match pe
            case AddOperator(a, b, outputs) => {
                assert a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs == pe.Outputs();
                assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
            }

            case SelectOperator(d, a, b, outputs) => {
                assert d in pe.WaitingInputs() && a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs == pe.Outputs();
                assert d == pe.Inputs()[0] && a == pe.Inputs()[1] && b == pe.Inputs()[2];
            }

            case OrderOperator(a, b, outputs) => {
                assert a in pe.WaitingInputs() && b in pe.WaitingInputs();
                assert outputs <= pe.Outputs();
                assert a == pe.Inputs()[0] && b == pe.Inputs()[1];
            }

            case _ => {}
    }

    lemma FirePEConfluence(idx1: PEIndex, idx2: PEIndex)
        requires WellFormed()
        requires idx1 != idx2
        requires 0 <= idx1 < |processingElements|
        requires 0 <= idx2 < |processingElements|
        requires IsFireable(idx1)
        requires IsFireable(idx2)

        ensures FirePE(idx1).WellFormed()
        ensures FirePE(idx2).WellFormed()
        ensures FirePE(idx1).IsFireable(idx2)
        ensures FirePE(idx2).IsFireable(idx1)
        ensures FirePE(idx1).FirePE(idx2) == FirePE(idx2).FirePE(idx1)
    {
        CommutableFiring(idx1, idx2);
        CommutableFiring(idx2, idx1);
        
        var fired1 := FirePE(idx1);
        var fired2 := FirePE(idx2);
        var fired12 := FirePE(idx1).FirePE(idx2);
        var fired21 := FirePE(idx2).FirePE(idx1);

        var pe1 := processingElements[idx1];
        var pe2 := processingElements[idx2];

        var inputs1 := pe1.WaitingInputs();
        var inputs2 := pe2.WaitingInputs();
        var outputs1 := pe1.WaitingOutputs();
        var outputs2 := pe2.WaitingOutputs();

        assert fired12 == fired21 by {
            reveal_WellFormed();
            reveal_FirePE();

            assert fired2.processingElements[idx1] == processingElements[idx1];
            assert fired1.processingElements[idx2] == processingElements[idx2];
            assert |fired1.channels| == |fired2.channels| == |fired12.channels| == |fired21.channels|;
            assert |fired1.processingElements| == |fired2.processingElements| == |fired12.processingElements| == |fired21.processingElements|;

            forall i | i in inputs1
                ensures fired2.channels[i][0] == channels[i][0]
            {
                FirePEChannelChange(idx2, i);
            }

            forall i | i in inputs2
                ensures fired1.channels[i][0] == channels[i][0]
            {
                FirePEChannelChange(idx1, i);
            }

            assert fired12.channels == fired21.channels by {
                forall i | 0 <= i < |fired12.channels|
                    ensures fired12.channels[i] == fired21.channels[i]
                {
                    FirePEChannelChange(idx1, i);
                    FirePEChannelChange(idx2, i);
                    fired2.FirePEChannelChange(idx1, i);
                    fired1.FirePEChannelChange(idx2, i);

                    if i in inputs1 && i in outputs1 {
                        // i is a self-feeding channel
                        FirePEDependency(this, fired2, idx1, i);
                        assert fired21.channels[i][..|fired21.channels[i]| - 1] == fired1.channels[i][..|fired1.channels[i]| - 1];
                        assert fired21.channels[i][|fired21.channels[i]| - 1] == fired1.channels[i][|fired1.channels[i]| - 1];

                    } else if i in inputs2 && i in outputs2 {
                        // Symmetric to the case above
                        FirePEDependency(this, fired1, idx2, i);
                        assert fired12.channels[i][..|fired21.channels[i]| - 1] == fired2.channels[i][..|fired2.channels[i]| - 1];
                        assert fired12.channels[i][|fired21.channels[i]| - 1] == fired2.channels[i][|fired2.channels[i]| - 1];

                    } else if i in inputs1 && i in outputs2 {
                        // idx2 sends output via i to idx1
                        calc {
                            fired12.channels[i][..|fired12.channels[i]| - 1];
                            fired1.channels[i];
                            channels[i][1..];
                            fired2.channels[i][1..|fired2.channels[i]| - 1];
                            fired21.channels[i][..|fired21.channels[i]| - 1];
                        }

                        FirePEDependency(this, fired1, idx2, i);
                        assert fired12.channels[i][|fired12.channels[i]| - 1] == fired21.channels[i][|fired21.channels[i]| - 1];

                        calc {
                            fired12.channels[i];
                            fired12.channels[i][..|fired12.channels[i]| - 1] + [fired12.channels[i][|fired12.channels[i]| - 1]];
                            fired21.channels[i][..|fired21.channels[i]| - 1] + [fired21.channels[i][|fired21.channels[i]| - 1]];
                            fired21.channels[i];
                        }

                    } else if i in inputs2 && i in outputs1 {
                        // symmetric to the case above
                        calc {
                            fired21.channels[i][..|fired12.channels[i]| - 1];
                            fired1.channels[i][1..|fired1.channels[i]| - 1];
                            channels[i][1..];
                            fired2.channels[i];
                            fired21.channels[i][..|fired21.channels[i]| - 1];
                        }

                        FirePEDependency(this, fired2, idx1, i);
                        assert fired21.channels[i][|fired21.channels[i]| - 1] == fired12.channels[i][|fired12.channels[i]| - 1];

                        calc {
                            fired21.channels[i];
                            fired21.channels[i][..|fired21.channels[i]| - 1] + [fired21.channels[i][|fired21.channels[i]| - 1]];
                            fired12.channels[i][..|fired12.channels[i]| - 1] + [fired12.channels[i][|fired12.channels[i]| - 1]];
                            fired12.channels[i];
                        }

                    } else if i in inputs1 && i !in outputs1 && i !in outputs2 {
                        // i is only an input channel of idx1 but not used for anything else
                        assert fired12.channels[i] == fired21.channels[i];

                    } else if i in inputs2 && i !in outputs1 && i !in outputs2 {
                        // i is only an input channel of idx2 but not used for anything else
                        assert fired12.channels[i] == fired21.channels[i];

                    } else if i in outputs1 && i !in inputs1 && i !in inputs2 {
                        // i is only an output channel of idx1 but not used for anything else
                        FirePEDependency(this, fired2, idx1, i);
                        assert fired12.channels[i] == fired21.channels[i];

                    } else if i in outputs2 && i !in inputs1 && i !in inputs2 {
                        // i is only an output channel of idx2 but not used for anything else
                        FirePEDependency(this, fired1, idx2, i);
                        assert fired12.channels[i] == fired21.channels[i];

                    } // else i is not changed at all
                }
            }

            assert fired12.processingElements == fired21.processingElements by {
                assert fired1.processingElements[idx1] == fired21.processingElements[idx1];
                assert fired2.processingElements[idx2] == fired12.processingElements[idx2];
            }
        }
        
        assert FirePE(idx1).FirePE(idx2) == FirePE(idx2).FirePE(idx1);
    }
}
