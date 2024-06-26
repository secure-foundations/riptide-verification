# Paper 282 OOPSLA 2024 Artifact Evaluation

# Introduction

This is the artifact for our paper *FlowCert: Translation Validation for Asynchronous Dataflow Programs via Dynamic Fractional Permissions*. The artifact is a translation validation tool for the RipTide compiler, which verifies that a given instance of compilation from an LLVM program to a dataflow program (on the RipTide CGRA architecture) is correct.

Besides the implementation itself, the artifact also includes: 1) a formalization of all theorems Section 5.1 in the paper in Verus, an SMT-based verification language; 2) test programs in our evaluation section (Section 6) and a script to reproduce the results.

# Hardware Dependencies

We have packed everything into two docker containers (one for x86_64 and one for ARM64). Please make sure that there is at least 10 GB of free disk space and 16 GB of memory.

The artifact has been tested on the following set ups:

- M1 Macbook Pro
- …TODO

# Getting Started Guide

Please follow the steps below to load and start our docker image.

1. Download the suitable Docker image for your machine.
2. Follow this guide to install Docker in your system: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/). We have tested on Docker version 20.10.24.
3. Load the image by `docker image load <docker image>`, which should add a new image tagged `flowcert` to your Docker environment.
4. Run the image by `docker run -it flowcert`.

# Step-by-Step Instructions

There are two claims in the paper to be verified in this artifact:

1. Section 5.1 is formalized in Verus.
2. Evaluation results and the bugs found.

### Formalization of Section 5.1

You can run `cd /build/flowcert/confluence && verus main.rs --rlimit 50` to verify the proofs of results in Section 5.1. This will take roughtly 20 seconds to finish.

We list how specifications and proofs in our Verus formalization (`/build/flowcert/confluence`) correspond to definitions and theorems in the paper. In Verus, functions declared with `spec fn` are definitions or specifications that need to be trusted; while proof functions declared with `proof fn` do not need to be trusted.

- `semantics.rs` defines the semantics of dataflow programs:
    - `struct Configuration` defines the configuration/execution state of a dataflow program;
    - `enum Operator` defines the types of operators (where non-memory operators like Carry, Steer, or Add are all categorized into `NonMemory`);
    - `spec fn state_inputs` and `spec fn state_outputs` are uninterpreted functions indicating the input and output channels of each operator;
    - `spec fn state_computation` denotes the exact semantics of each non-memory operator, which we left uninterpreted to allow arbitrary non-memory operator semantics.
    - `spec fn step` defines how to fire one operator in a configuration
- `permission.rs` corresponds to Section 5.1:
    - `struct Permission` defines a permission algebra (Definition 8) with `k` left uninterpreted as `spec fn permission_write_split`.
    - `spec fn Permission::disjoint` corresponds to Definition 9.
    - `struct AugmentedConfiguration` corresponds to Definition 10.
    - `spec fn AugmentedConfiguration::valid` corresponds to Definition 11.
    - `spec fn consistent_step` corresponds to Definition 12.
    - `spec fn AugmentedTrace::valid` corresponds to Definition 13.
    - `proof fn lemma_consistent_steps_commute` proves Lemma 1.
    - `proof fn lemma_consistent_trace_commutes` proves Lemma 2.
    - `proof fn theorem_bounded_confluence` proves Theorem 1.

### Evaluation Results

1. Run `cd /build/flowcert/evaluations && make -j <number of cores>` to run FlowCert on all examples in Figure 5. This will take a few minutes to finish depending on the number of cores you have. The bugs indicated in Figure 5 are already fixed in the RipTide compiler included in the artifact, so there should not be any errors.
2. Run `python3 summarize.py` to print out a LaTeX table corresponding to Figure 5.

# Reusability Guide

Our FlowCert tool should be reusable for compiling other C functions using the RipTide compiler.

For example, take this simple C function that computes the dot product of vectors `A` and `B` both of length `len`:

```bash
void f(int *A, int *B, int len, int *result)
{
    int p = 0;
    for (int i = 0; i < len; i++) {
        p += A[i] * B[i];
    }
    *result = p;
}
```

To compile and validate the compilation using FlowCert:

1. Make sure you are in the `/build/flowcert` directory.
2. Save the function to a file `test.c` by

    ```bash
    cat << EOF > test.c
    void f(int *A, int *B, int len, int *result)
    {
        int p = 0;
        for (int i = 0; i < len; i++) {
            p += A[i] * B[i];
        }
        *result = p;
    }
    EOF
    ```

3. Run

    ```
    python3 -m tools.sdc \
            --lib-dc $LIBDC_PATH \
            --llvm-bin $LLVM_12_BIN \
    			  --bisim \
    			  --bisim-permission-unsat-core \
    			  --bisim-cut-point-expansion \
    			  --bisim-permission-fractional-reads 4 \
    			  test.c
    ```

    (`LIBDC_PATH` and `LLVM_12_BIN` are environment variables available in the Docker image)

4. If nothing goes wrong, the logs should contain both `bisim check succeeds` and `confluence check result: sat - confluent`.
