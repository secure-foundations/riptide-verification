
"""
1. init state
2. loop state
    channels:
    1: empty

3. end state



"""

from collections import OrderedDict

from semantics import smt

from semantics.matching import *
from semantics.llvm.parser import Parser
from semantics.llvm.semantics import Configuration, NextConfiguration, FunctionReturn


def main():
    with open("examples/bisim/test-1.ll") as f:
        module = Parser.parse_module(f.read())

    function = module.functions["@test"]

    loop_header_1 = Configuration(
        module,
        function,
        current_block="header",
        previous_block="entry",
        current_instr_counter=0,
        variables=OrderedDict(),
        path_conditions=[],
    )

    loop_header_2 = Configuration(
        module,
        function,
        current_block="header",
        previous_block="body",
        current_instr_counter=0,
        variables=OrderedDict([
            (r"%A", smt.FreshSymbol(smt.BVType(64), "A_%d")),
            (r"%len", smt.FreshSymbol(smt.BVType(32), "len_%d")),
            (r"%inc", smt.FreshSymbol(smt.BVType(32), "inc_%d")),
        ]),
        path_conditions=[],
    )

    end_pattern = Configuration(
        module,
        function,
        current_block="end",
        previous_block="header",
        current_instr_counter=0,
        variables=OrderedDict(),
        path_conditions=[],
    )

    # queue = [Configuration.get_initial_configuration(module, function)]
    queue = [loop_header_2.copy()]
    final_state_count = 0

    while len(queue):
        config = queue.pop(0)
        results = config.step()

        for result in results:
            if isinstance(result, NextConfiguration):
                # print(result.config)
                # match = loop_header_1.match(result.config)
                # if isinstance(match, MatchingSuccess):
                #     print("found successul matching to loop header 1")
                #     print("substitution: " + str(match.substitution))
                #     print("condition: " + str(match.condition.simplify()))
                #     # continue

                match = loop_header_2.match(result.config)
                if isinstance(match, MatchingSuccess):
                    print("### matching success at loop header 2")
                    print(result.config)
                    print("substitution: " + str(match.substitution))
                    print("condition: " + str(match.condition.simplify()))
                    continue

                queue.append(result.config)

            elif isinstance(result, FunctionReturn):
                final_state_count += 1
                print(f"\nfinal config #{final_state_count}")
                print(f"returns {result.value}, {result.final_config}")
            else:
                assert False, f"unsupported result {result}"
 

if __name__ == "__main__":
    main()
