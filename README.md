FlowCert: Translation Validation for RipTide
---

This project is a translation validation tool for the [RipTide](https://doi.org/10.1109/MICRO56248.2022.00046)
dataflow compiler: given an input LLVM function and the output
dataflow program from the compiler, we check that they are equivalent,
on all possible inputs and schedules of dataflow operators.

The tool operates in two steps:
1. Prove that the LLVM program is bisimilar to the dataflow program, on a canonical schedule of operators
2. Prove the confluence of the dataflow program via linear permission tokens (a dynamic version of fractional permissions).

## Usage

Coming soon.
