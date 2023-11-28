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

Most command-line tools are in Python modules `tools.*`.
Run `python3 -m pip install -r requirements.txt` before using any of the tools below.

### `tools.sdc`: A wrapper for RipTide's standalone dataflow compiler with translation validation

Coming soon.

### `tools.run_llvm`: A symbolic executor for LLVM

To symbolically execute a function in a given LLVM IR file, run
```
python3 -m tools.run_llvm <input LLVM IR source>
```
This will do a breadth first search on the state space (which may not terminate) and output final symbolic states.

### `tools.run_dataflow`: A symbolic executor for dataflow program with bounded confluence check

```
python3 -m tools.run_dataflow <o2p file>
```

### `tools.visualizer`

To generate a DOT description of the dataflow program, run
```
python3 -m tools.visualizer <o2p file>
```
