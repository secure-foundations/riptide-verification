import re
import json


def main():
    tests = [
        "bfs",
        "dconv",
        "dfs",
        "dmm",
        "dmv",
        "fft",
        "sconv",
        "smm",
        "smv",
        "sort",
        "nn_conv",
        "nn_fc",
        "nn_norm",
        "nn_pool",
        "nn_relu",
        "nn_vadd",
        "Dither",
        "SpMSpMd",
        "SpMSpVd",
        "SpSlice",
    ]

    for test in tests:
        with open(f"{test}.{test}.lso.ll") as f:
            llvm_loc = len(f.readlines())

        with open(f"{test}.c") as f:
            c_loc = len(f.readlines())

        with open(f"{test}.{test}.o2p") as f:
            num_operators = len(json.load(f)["vertices"])

        with open(f"{test}.bisim.out") as f:
            content = f.read()
            num_cut_points = content.count("[simulation] llvm cut point") + 1

            match = re.search(r"\[simulation\] bisim check took ((\d+)\.(\d+))s", content)
            assert match is not None
            bisim_time = float(match.group(1))

            match = re.search(r"\[simulation\] confluence check took ((\d+)\.(\d+))s", content)
            assert match is not None
            confluence_time = float(match.group(1))

            match = re.search(r"\[simulation\] checking sat of (\d+) memory permission constraints", content)
            assert match is not None
            num_perm_constraints = int(match.group(1))

        test_name = test.replace("_", "\\textunderscore ")
        print(f"\\texttt{{{test_name}}} & {c_loc} & {num_operators} & {num_perm_constraints} & {bisim_time} & {confluence_time} \\\\")


if __name__ == "__main__":
    main()
