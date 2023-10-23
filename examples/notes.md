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



# Generating dataflow cut points

Come up with the LLVM cut points first (which is relatively easy).
Then we run these cut points.

For each cut point i, and each other cut point j, we would have a list of configurations

c_i_j_1 /\ phi_i_j_1 , ..., c_i_j_k /\ phi_i_j_k

matched to cut point j (starting from cut point i).

For each of them, we get the following information in addition to the final state:
- The trace (list of LLVM instructions executed)
- Use count of variables
  * which user of a defined variable is actually executed, this can help us figure out which channel should be left empty or not

Observation:
- The path from definition -> use should only have been gated with T/F or Inv (potentially multiple of them)


Regarding state of the channels:
- Each path from a non-(steer/inv) gate to another should correspond to a def->use edge in the LLVM program (def could be a constant).


For each LLVM cut point i, generate a dataflow cut point i
- Operator state:
    - All carry gates for the current and all parent loop headers are set to `loop` state
    - All invariant gates for the current and all parent loop headers are set to `loop` state (and attached with an invariant value)
    - All other gates are set to `start` state
- Channel states:
    - From any node N to a non-<steer/inv> node M: go back to find the actual producer of N (i.e.  a non-<steer/inv> node). If the def-use edge is not executed in the LLVM program, put a single symbolic value in the channel
    - From a decider to carry/inv gate:
        * empty if not related to the current loop
        * a single true value if it's in the current loop or any parent loop
    - From a constant to a node:
        * empty if used in the LLVM program
        * a single constant value otherwise
    - Otherwise: empty


Plan:
1. Get the LLVM trace thing
2. Use the trace and the rules above to generate dataflow cut points

Updated plan:
- We run the dataflow config first and then figure out the invariant states:
    - The initial state is always fixed, so we start from that
    - We follow the LLVM schedule, and that should get us to another invariant state (but maybe not all)
    - We infer the actual template from the resulting dataflow state
    - And then we start from that state again -- until we cover all invariant state
    - All invariant state should be reachable from the initial state (otherwise the invariant state is useless)
