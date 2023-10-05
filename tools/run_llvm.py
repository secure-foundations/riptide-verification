import argparse

from semantics.llvm.parser import Parser
from semantics.llvm.semantics import Configuration, NextConfiguration, FunctionReturn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("llvm_ir_file", help="LLVM IR file (.ll)")
    args = parser.parse_args()

    with open(args.llvm_ir_file) as f:
        module = Parser.parse_module(f.read())

    print(module)

    assert(len(module.functions) == 1)

    config = Configuration.get_initial_configuration(module, list(module.functions.values())[0])

    print(config)

    queue = [config]
    final_state_count = 0

    while len(queue):
        config = queue.pop(0)
        results = config.step()
        for result in results:
            if isinstance(result, NextConfiguration):
                queue.append(result.config)
            elif isinstance(result, FunctionReturn):
                final_state_count += 1
                print(f"\nfinal config #{final_state_count}")
                print(f"returns {result.value}, {result.final_config}")
            else:
                assert False, f"unsupported result {result}"


if __name__ == "__main__":
    main()
