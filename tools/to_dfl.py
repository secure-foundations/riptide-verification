import json
import argparse

from semantics.dataflow import (
    DataflowGraph, NextConfiguration, StepException,
    Configuration, WORD_WIDTH, ProcessingElement,
    FunctionArgument, ConstantValue,
)


def get_pe_io(pe: ProcessingElement) -> str:
    return f"{', '.join(['in C' + str(input.id) for input in pe.inputs] + ['out C' + str(output.id) for outputs in pe.outputs.values() for output in outputs])}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("o2p", help="Input o2p file")
    args = parser.parse_args()

    with open(args.o2p) as dataflow_source:
        dfg = DataflowGraph.load_dataflow_graph(json.load(dataflow_source))

    # function arguments:
    #   *int => mutable [int]
    #   int => const int



    for function_arg in dfg.function_arguments:
        if function_arg.typ == "i32":
            print(f"const {function_arg.variable_name}: int")
        elif function_arg.typ == "i32*":
            print(f"mut {function_arg.variable_name}: [int]")
        else:
            assert False, f"unsupported argument type {function_arg.typ}"

    print("")

    for channel in dfg.channels:
        print(f"perm PV_C{channel.id}(int)")
        print(f"chan C{channel.id}: int | PV_C{channel.id}(C{channel.id})")

    proc_names = []

    for pe in dfg.vertices:
        print("")
        if pe.operator == "CF_CFG_OP_CARRY":
            name1 = f"CarryI{pe.id}"
            name2 = f"CarryS{pe.id}"
            proc_names.append(name1)

            decider, a, b = pe.inputs

            print(f"perm PV_{name1}()")
            print(f"perm PV_{name2}()")

            print(f"proc {name1} | PV_{name1}(), {get_pe_io(pe)} =")
            print(f"  recv a <- C{a.id}")
            for out in pe.outputs[0]:
                print(f"  send a -> C{out.id}")
            print(f"  {name2}")
            print()

            print(f"proc {name2} | PV_{name2}(), {get_pe_io(pe)} =")
            print(f"  recv d <- C{decider.id}")
            print(f"  if not (d = 0) then")
            print(f"    recv b <- C{b.id}")
            for out in pe.outputs[0]:
                print(f"    send b -> C{out.id}")
            print(f"    {name2}")
            print(f"  else")
            print(f"    {name1}")
            print(f"  end")

        elif pe.operator == "ARITH_CFG_OP_ULT":
            name = f"Ult{pe.id}"
            proc_names.append(name)

            print(f"perm PV_{name}()")

            a, b = pe.inputs
            print(f"proc {name} | PV_{name}(), {get_pe_io(pe)} =")
            print(f"  recv a <- C{a.id}")
            print(f"  recv b <- C{b.id}")
            print(f"  if a < b then")
            for out in pe.outputs[0]:
                print(f"    send 1 -> C{out.id}")
            print(f"    {name}")
            print(f"  else")
            for out in pe.outputs[0]:
                print(f"    send 0 -> C{out.id}")
            print(f"    {name}")
            print(f"  end")

        elif pe.operator == "CF_CFG_OP_STEER":
            name = f"Steer{pe.id}"
            proc_names.append(name)

            print(f"perm PV_{name}()")

            decider, a = pe.inputs
            print(f"proc {name} | PV_{name}(), {get_pe_io(pe)} =")
            print(f"  recv d <- C{decider.id}")
            print(f"  recv a <- C{a.id}")
            print(f"  if not (d = 0) then")
            for out in pe.outputs[0]:
                print(f"    send a -> C{out.id}")
            print(f"    {name}")
            print(f"  else")
            print(f"    {name}")
            print(f"  end")

        elif pe.operator == "MEM_CFG_OP_STORE":
            name = f"Store{pe.id}"
            proc_names.append(name)

            print(f"perm PV_{name}()")

            base, index, value, *rest = pe.inputs
            assert isinstance(base.constant, FunctionArgument), "unsupported non-constant base value"

            print(f"proc {name} | PV_{name}(), {get_pe_io(pe)} =")
            print(f"  recv index <- C{index.id}")
            print(f"  recv value <- C{value.id}")
            print(f"  write value -> {base.constant.variable_name}[index]")
            if len(pe.outputs) != 0:
                for out in pe.outputs[0]:
                    print(f"  send 0 -> C{out.id}")
            print(f"  {name}")

        elif pe.operator == "ARITH_CFG_OP_ADD":
            name = f"Add{pe.id}"
            proc_names.append(name)

            print(f"perm PV_{name}()")

            a, b = pe.inputs
            print(f"proc {name} | PV_{name}(), {get_pe_io(pe)} =")
            print(f"  recv a <- C{a.id}")
            print(f"  recv b <- C{b.id}")
            for out in pe.outputs[0]:
                print(f"  send a + b -> C{out.id}")
            print(f"  {name}")
        else:
            assert False, f"unsupported operator {pe.operator}"

    # Finally, generate a whole program process
    for channel in dfg.channels:
        if channel.constant is not None and channel.hold and not (isinstance(channel.constant, FunctionArgument) and channel.constant.typ != "i32"):
            print()
            print(f"proc Hold{channel.id} | out C{channel.id} =")

            if isinstance(channel.constant, FunctionArgument):
                if channel.constant.typ == "i32":
                    print(f"  send {channel.constant.variable_name} -> C{channel.id}")
            elif isinstance(channel.constant, ConstantValue):
                print(f"  send {channel.constant.value} -> C{channel.id}")
            else:
                assert False, f"unsupported constant {channel.constant}"

            print(f"  Hold{channel.id}")

    print()
    print(f"proc Program | all = {' || '.join(proc_names)} ||")
    for channel in dfg.channels:
        if channel.constant is not None:
            if channel.hold and not (isinstance(channel.constant, FunctionArgument) and channel.constant.typ != "i32"):
                print(f"  Hold{channel.id} ||")

    for channel in dfg.channels:
        if channel.constant is not None and not channel.hold:
            if isinstance(channel.constant, FunctionArgument):
                if channel.constant.typ == "i32":
                    print(f"  send {channel.constant.variable_name} -> C{channel.id}")
            elif isinstance(channel.constant, ConstantValue):
                print(f"  send {channel.constant.value} -> C{channel.id}")
            else:
                assert False, f"unsupported constant {channel.constant}"


if __name__ == "__main__":
    main()