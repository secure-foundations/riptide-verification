# This is a wrapper around https://github.com/CMUAbstract/sdc

from typing import Optional, Iterable
from dataclasses import dataclass

import os
import re
import shlex
import shutil
import tempfile
import argparse
import subprocess

import semantics.llvm as llvm

from utils import logging

from .bisim import run_bisim

logger = logging.getLogger(__name__)


@dataclass
class Environment:
    lib_dc_path: str
    clang_path: str
    opt_path: str
    llvm_version: str


def get_clang_major_version(clang_path: str) -> Optional[int]:
    output = subprocess.check_output([ clang_path, "--version" ]).decode()
    match = re.search(r"clang version ([0-9]+)\.[0-9]+(\.[0-9]+)?", output.splitlines()[0])
    if match is None:
        return None
    return int(match.group(1))


def get_opt_major_version(opt_path: str) -> Optional[int]:
    output = subprocess.check_output([ opt_path, "--version" ]).decode()
    match = re.search(r"LLVM version ([0-9]+)\.[0-9]+(\.[0-9]+)?", output)
    if match is None:
        return None
    return int(match.group(1))


def find_llvm(lib_dc_path: Optional[str] = None, llvm_bin_path: Optional[str] = None) -> Environment:
    """
    Find paths to libDC, clang, and opt and check for suitable versions
    """

    lib_dc_path = lib_dc_path or os.environ.get("LIB_DC_PATH")
    assert lib_dc_path is not None, "failed to find libDC shared library (either from --lib-dc flag or LIB_DC_PATH environment)"
    assert os.path.isfile(lib_dc_path), f"libDC file {lib_dc_path} does not exist"
    lib_dc_path = os.path.realpath(lib_dc_path)

    llvm_bin_dir = llvm_bin_path or os.environ.get("LLVM_BIN_PATH")
    if llvm_bin_dir is None:
        clang_path = shutil.which("clang")
        opt_path = shutil.which("opt")
        assert clang_path is not None and opt_path is not None, \
               "failed to find clang or opt in the environment, please set --llvm-bin or use environment variable LLVM_BIN_PATH"
    else:
        assert os.path.isdir(llvm_bin_dir), f"LLVM bin path {llvm_bin_dir} does not exist"
        llvm_bin_dir = os.path.realpath(llvm_bin_dir)
        clang_path = os.path.join(llvm_bin_dir, "clang")
        opt_path = os.path.join(llvm_bin_dir, "opt")
        assert os.path.isfile(clang_path) and os.path.isfile(opt_path), \
               "failed to find clang or opt in the environment"

    clang_version = get_clang_major_version(clang_path)
    opt_version = get_opt_major_version(opt_path)

    assert clang_version == opt_version, f"clang and opt version mismatch ({clang_version} vs {opt_version})"
    assert clang_version == 12 or clang_version == 16, f"currently only supports LLVM version 12 or 16 ({clang_version} given)"

    return Environment(lib_dc_path, clang_path, opt_path, clang_version)


def check_dep(dependencies: Iterable[str], results: Iterable[str]) -> bool:
    """
    Check that all files in results are younger than files in dependencies
    """
    satisfied = True
    min_result_timestamp = float("inf")

    for result in results:
        # should run the target if any of the results does not exist
        if not os.path.exists(result):
            satisfied = False
            break
        else:
            stat = os.stat(result)
            min_result_timestamp = min(min_result_timestamp, stat.st_mtime)
    else:
        # should also run the target if any of the dependencies changed
        # after some of the results were modified
        for path in dependencies:
            assert os.path.exists(path), "missing dependency {}".format(path)
            stat = os.stat(path)
            if stat.st_mtime > min_result_timestamp:
                satisfied = False
                break

    return satisfied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Input source file (.c or .ll)")
    parser.add_argument("--lib-dc", help="Path to the libDC shared library (usually named libDC.{{so,dylib}})")
    parser.add_argument("--llvm-bin", help="Path to the LLVM and Clang binaries")

    parser.add_argument("--normal", action="store_const", const=True, default=False, help="Disable some flags used for bisim, allow optimizations such as streamify")
    parser.add_argument("--gen-norm-ll", action="store_const", const=True, default=False, help="Generate LLVM code after some normalizations (.norm.ll)")
    parser.add_argument("--gen-lso-ll", action="store_const", const=True, default=False, help="Generate LLVM code after lso ordering (.lso.ll)")
    parser.add_argument("--gen-log", action="store_const", const=True, default=False, help="Generate compilation log (.log)")

    parser.add_argument("--bisim", action="store_const", const=True, default=False,
                        help="Enable bisimulation check after execution (this automatically disables --normal and enables --gen-lso-ll)")
    parser.add_argument("--bisim-permission-fractional-reads", default=4, type=int, help="How many read permissions a write permission can be split into (for --bisim)")
    parser.add_argument("--bisim-permission-unsat-core", action="store_const", const=True, default=False, help="Output unsat core from the permission solver if failed (for --bisim)")
    parser.add_argument("--bisim-cut-point-expansion", action="store_const", const=True, default=False, help="Enable cut point expansion for confluence checking (for --bisim)")

    logging.add_arguments(parser)
    args = parser.parse_args()
    logging.basic_config(args)

    if args.bisim:
        args.gen_lso_ll = True
        assert not args.normal, "option --bisim cannot be used with --normal: some options enabled with --normal is not yet supported by the bisimulation checker"

    input_file_name = args.source_file
    assert input_file_name.endswith(".c") or input_file_name.endswith(".ll"), \
           f"expecting file extension .c or .ll, {input_file_name} given"
    input_file_name = os.path.realpath(input_file_name)
    assert os.path.isfile(input_file_name)

    env = find_llvm(args.lib_dc, args.llvm_bin)
    logger.info(f"using libDC {env.lib_dc_path}")
    logger.info(f"using clang {env.clang_path} (version {env.llvm_version})")
    logger.info(f"using opt {env.opt_path} (version {env.llvm_version})")

    tmp_dir = tempfile.mkdtemp()
    delete = True

    try:
        if input_file_name.endswith(".c"):
            base_name = os.path.basename(input_file_name)[:-2]
        else:
            assert input_file_name.endswith(".ll")
            base_name = os.path.basename(input_file_name)[:-3]

        tmp_base_path = os.path.join(tmp_dir, base_name)

        if input_file_name.endswith(".c"):
            ll_path = tmp_base_path + ".ll"
            norm_ll_path = tmp_base_path + ".norm.ll"

            # clang -Oz -S -m32 -emit-llvm <C_FILE> -o <LL_FILE>
            clang_log_path = tmp_base_path + ".clang.log"
            with open(clang_log_path, "wb") as clang_log_file:
                cmd = [
                    env.clang_path,
                    "-Oz",
                    # "-DALIAS_RESTRICT", NOTE: might be used in the RipTide benchmark
                    "-m32", # RipTide word width is 32-bit
                    "-S",
                    "-emit-llvm",
                    input_file_name,
                    "-o",
                    ll_path,
                ]
                logger.info(f"running {shlex.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    stdout=clang_log_file,
                    stderr=clang_log_file,
                )

            if args.gen_log or result.returncode != 0:
                shutil.copy(clang_log_path, os.path.dirname(input_file_name))

            assert result.returncode == 0, f"clang failed with exit code {result.returncode}"

            # opt --enable-new-pm=0 -S -loop-simplify -simplifycfg -lcssa -mem2reg -loop-simplify <LL_FILE> -o <NORM_LL_FILE>
            opt_log_path = tmp_base_path + ".opt.log"
            with open(opt_log_path, "wb") as opt_log_file:
                cmd = [
                    env.opt_path,
                    "--enable-new-pm=0",
                    "-S",
                    "-loop-simplify",
                    "-simplifycfg",
                    "-lcssa",
                    "-mem2reg",
                    "-loop-simplify",
                    ll_path,
                    "-o",
                    norm_ll_path,
                ]
                logger.info(f"running {shlex.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    stdout=opt_log_file,
                    stderr=opt_log_file,
                )

            if args.gen_log or result.returncode != 0:
                shutil.copy(opt_log_path, os.path.dirname(input_file_name))

            assert result.returncode == 0, f"opt failed with exit code {result.returncode}"

            if args.gen_norm_ll:
                shutil.copy(norm_ll_path, os.path.dirname(input_file_name))

        else:
            norm_ll_path = input_file_name

        with open(norm_ll_path) as norm_ll_file:
            llvm_module = llvm.Parser.parse_module(norm_ll_file.read())

        for function_name in llvm_module.functions.keys():
            logger.info(f"compiling function {function_name}")

            lso_ll_name = base_name + "." +  function_name[1:] + ".lso.ll"
            o2p_name = base_name + "." +  function_name[1:] + ".o2p"

            sdc_log_path = os.path.join(tmp_dir, base_name + "." + function_name[1:] + ".sdc.log")
            lso_ll_path = os.path.join(tmp_dir, lso_ll_name)
            o2p_path = os.path.join(tmp_dir, o2p_name)

            final_lso_ll_path = os.path.join(os.path.dirname(input_file_name), lso_ll_name)
            final_o2p_path = os.path.join(os.path.dirname(input_file_name), o2p_name)

            # If input file has not changed since the last time the .o2p and .lso.ll files are generated
            # we skip the sdc call (because its output is non-deterministic)
            if not check_dep([ input_file_name, env.lib_dc_path, __file__ ], [ final_o2p_path, final_lso_ll_path ]):
                with open(sdc_log_path, "wb") as sdc_log_file:
                    cmd = [
                        env.opt_path,
                        "-S",
                        "-load", env.lib_dc_path,
                        "-DC",
                        "-func", function_name,
                        "-json", o2p_path,
                        "-lso-ll-out", lso_ll_path,
                        norm_ll_path,
                    ]
                    if not args.normal:
                        cmd += [
                            "-fno-hold-channel",
                            "-fno-stream",
                            "-fno-dedup",
                            "-additional-id-lcssa",
                            "-fno-array-dep",
                        ]

                    logger.info(f"running {shlex.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        stdout=sdc_log_file,
                        stderr=sdc_log_file,
                    )

                if args.gen_log or result.returncode != 0:
                    shutil.copy(sdc_log_path, os.path.dirname(input_file_name))

                assert result.returncode == 0, f"sdc failed with exit code {result.returncode}"
                if args.gen_lso_ll:
                    shutil.copy(lso_ll_path, final_lso_ll_path)
                shutil.copy(o2p_path, final_o2p_path)
            else:
                logger.info("skipping sdc")

            if args.bisim:
                logger.info(f"running bisim check for function {function_name}")
                run_bisim(
                    final_o2p_path,
                    final_lso_ll_path,
                    function_name,
                    args.bisim_permission_fractional_reads,
                    args.bisim_permission_unsat_core,
                    args.bisim_cut_point_expansion,
                )

    except Exception as e:
        delete = False
        raise e

    finally:
        if delete:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
