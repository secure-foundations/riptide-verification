WaveCert: Translation Validation for RipTide
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

WaveCert requires the RipTide dataflow compiler to work, which is currently not public yet.
But assuming the dataflow compiler (sdc) is available as a shared library `<LIBDC>` (with extension `.so`/`.dylib`),
and LLVM 12.0.0 binaries are located in the directory `<LLVM_BIN>`, then we can use it via:
```
python3 -m tools.sdc --lib-dc <LIBDC> --llvm-bin <LLVM_BIN> --gen-lso-ll --bisim <input .c file>
```
which will compile all functions in the input C file to dataflow graphs, and also perform translation validation after each compilation
to check the correctness of compilation.

If the script succeeds without exception and the output of the script includes these lines
```
bisim check succeeds
```
```
confluence check result: sat - confluent
```
Then the tool verifies that the compilation is correct (in our model of dataflow graphs).

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
