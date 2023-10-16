1. We can partition the set of Carry and Inv gates by the loop it belongs to.
2. When are single-use constant value used? and at which cut points to pop them?
    * Constant value in a carry gate?
    * 
3. Invariant state for a loop:
    * For all carry gate, set state to CarryOperator.loop and put one true value in decider and a value in third input.
    * For all invariant gate: set state to InvariantOperator.loop with an invariant value. Put one true value in the decider channel
    * For any live variables at the loop header, set some value in their corresponding channels
    * This applies to all carry/invariant gates in the parent loops as well
    * edit: maybe not all live variables? just the variables live in any path from the target loop header to any other loop header (or cut points)
    * edit: in the schedule, run inv gates before their deciders?
    * edit: don't set the backedge value for carry gates of the outer loops?

