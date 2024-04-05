from __future__ import annotations
from typing import Tuple, Optional

from collections import OrderedDict
from dataclasses import dataclass, field


WORD_WIDTH = 32 # addresses are 32-bit to avoid translation from/to dataflow memory
BYTE_WIDTH = 8


class ASTNode:
    ...


@dataclass
class Module(ASTNode):
    functions: OrderedDict[str, Function]

    def resolve_uses(self):
        for function in self.functions.values():
            function.resolve_uses(self)

    def __str__(self) -> str:
        return "\n\n".join(str(function) for function in self.functions.values())


class Type(ASTNode):
    def get_bit_width(self) -> int:
        raise NotImplementedError()


@dataclass
class VoidType(Type):
    def __str__(self) -> str:
        return "void"


@dataclass
class IntegerType(Type):
    bit_width: int

    def __str__(self) -> str:
        return f"i{self.bit_width}"

    def get_bit_width(self) -> int:
        return self.bit_width


@dataclass
class PointerType(Type):
    base_type: Type

    def __str__(self) -> str:
        return f"{self.base_type}*"

    def get_bit_width(self) -> int:
        return WORD_WIDTH


@dataclass
class ArrayType(Type):
    base_type: Type
    num_elements: int

    def __str__(self) -> str:
        return f"[{self.num_elements} x {self.base_type}]"


@dataclass
class FunctionType(Type):
    return_type: Type
    parameter_types: Tuple[Type, ...]
    variable_args: bool

    def __str__(self) -> str:
        if self.variable_args:
            return f"{self.return_type} ({', '.join(tuple(map(str, self.parameter_types)) + ('...',))})"
        else:
            return f"{self.return_type} ({', '.join(map(str, self.parameter_types))})"


class Value(ASTNode):
    def get_type(self) -> Type:
        raise NotImplementedError()

    def resolve_uses(self, function: Function) -> Value:
        return self


@dataclass
class FunctionParameter(Value):
    type: Type
    name: str
    attributes: Tuple[str, ...]

    def get_type(self) -> Type:
        return self.type

    def is_noalias(self) -> bool:
        return "noalias" in self.attributes

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class Function(ASTNode):
    name: str
    return_type: str

    parameters: OrderedDict[str, FunctionParameter]
    blocks: OrderedDict[str, BasicBlock]
    definitions: OrderedDict[str, Instruction] = field(init=False)

    module: Optional[Module] = None

    def __post_init__(self):
        self.definitions = OrderedDict()

        # Find definition for each variable
        for block in self.blocks.values():
            for instruction in block.instructions:
                name = instruction.get_defined_variable()
                if name is not None:
                    assert name not in self.definitions, f"redefinition of {name}"
                    self.definitions[name] = instruction

    def resolve_uses(self, module: Module):
        self.module = module
        for block in self.blocks.values():
            for instruction in block.instructions:
                instruction.resolve_uses(self)

    def __str__(self) -> str:
        parameters_string = ", ".join(str(parameter) for parameter in self.parameters.values())
        blocks_string = "\n\n".join(str(block) for block in self.blocks.values())
        return f"{self.return_type} {self.name}({parameters_string}) {{\n{blocks_string}\n}}"

    def get_block(self, name: str) -> BasicBlock:
        assert name in self.blocks, f"cannot find block {name}"
        return self.blocks[name]


@dataclass
class BasicBlock(ASTNode):
    name: str
    instructions: Tuple[Instruction, ...]

    def __str__(self) -> str:
        instructions_string = "\n    ".join(instruction.get_full_string() for instruction in self.instructions)
        return f"  {self.name}:\n    {instructions_string}"

    def get_nth_instruction(self, n: int) -> Instruction:
        assert 0 <= n and n < len(self.instructions), f"index {n} out of range"
        return self.instructions[n]


# Values who types cannot be inferred yet
class UnresolvedValue(Value):
    def attach_type(self, typ: Type) -> Value:
        raise NotImplementedError()

    def resolve_uses(self, function: Function) -> Value:
        raise NotImplementedError()

@dataclass
class UnresolvedIntegerValue(UnresolvedValue):
    value: int

    def attach_type(self, typ: Type) -> IntegerConstant:
        assert isinstance(typ, IntegerType)
        return IntegerConstant(typ, self.value)


class UnresolvedNullValue(UnresolvedValue):
    def attach_type(self, typ: Type) -> NullConstant:
        assert isinstance(typ, PointerType)
        return NullConstant(typ.base_type)


class UnresolvedUndefValue(UnresolvedValue):
    def attach_type(self, typ: Type) -> UndefConstant:
        return UndefConstant(typ)


# Unresolved variable use
@dataclass
class UnresolvedVariable(UnresolvedValue):
    name: str
    type: Optional[Type] = None

    def attach_type(self, typ: Type) -> UnresolvedVariable:
        return UnresolvedVariable(self.name, typ)

    def resolve_uses(self, function: Function) -> Value:
        if self.name in function.parameters:
            return function.parameters[self.name]
        elif self.name in function.definitions:
            return function.definitions[self.name]
        else:
            assert False, f"unable to resolve variable {self.name} in function {function.name}"


class Constant(Value):
    ...


@dataclass
class IntegerConstant(Constant):
    type: IntegerType
    value: int

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.value}"


@dataclass
class NullConstant(Constant):
    base_type: Type

    def get_type(self) -> Type:
        return PointerType(self.base_type)

    def __str__(self) -> str:
        return f"{self.base_type}* null"


@dataclass
class UndefConstant(Constant):
    type: Type

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} undef"


class Instruction(Value):
    def get_defined_variable(self) -> Optional[str]:
        return None

    def resolve_uses(self, function: Function) -> None:
        ...

    def get_full_string(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.get_full_string()


@dataclass
class AddInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = add {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class SubInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = sub {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class MulInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = mul {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class AndInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = and {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class OrInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = or {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class XorInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = xor {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class ShlInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = shl {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class LshrInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = lshr {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class AshrInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = ashr {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class ZextInstruction(Instruction):
    name: str
    from_type: IntegerType
    to_type: IntegerType
    value: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.value = self.value.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = zext {self.from_type} {self.value} to {self.to_type}"

    def get_type(self) -> Type:
        return self.to_type

    def __str__(self) -> str:
        return f"{self.to_type} {self.name}"


@dataclass
class SextInstruction(Instruction):
    name: str
    from_type: IntegerType
    to_type: IntegerType
    value: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.value = self.value.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = sext {self.from_type} {self.value} to {self.to_type}"

    def get_type(self) -> Type:
        return self.to_type

    def __str__(self) -> str:
        return f"{self.to_type} {self.name}"


@dataclass
class IntegerCompareInstruction(Instruction):
    name: str
    cond: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = icmp {self.cond} {self.type}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return IntegerType(1)

    def __str__(self) -> str:
        return f"i1 {self.name}"


@dataclass
class SelectInstruction(Instruction):
    name: str
    cond: Value
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.cond = self.cond.resolve_uses(function)
        self.left = self.left.resolve_uses(function)
        self.right = self.right.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = select {self.cond}, {self.left}, {self.right}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class GetElementPointerInstruction(Instruction):
    name: str
    base_type: Type
    pointer: Value
    indices: Tuple[Value, ...]

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.pointer = self.pointer.resolve_uses(function)
        self.indices = tuple(index.resolve_uses(function) for index in self.indices)

    def get_full_string(self) -> str:
        indices_string = ", ".join(str(index) for index in self.indices)
        return f"{self.name} = getelementptr {self.base_type}, {self.pointer}, {indices_string}"

    def get_type(self) -> Type:
        assert len(self.indices) == 1, "unsupported"
        return PointerType(self.base_type)

    def __str__(self) -> str:
        return f"{self.get_type()} {self.name}"


@dataclass
class LoadInstruction(Instruction):
    name: str
    base_type: Type
    pointer: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.pointer = self.pointer.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"{self.name} = load {self.base_type}, {self.pointer}"

    def get_type(self) -> Type:
        return self.base_type

    def __str__(self) -> str:
        return f"{self.base_type} {self.name}"


@dataclass
class StoreInstruction(Instruction):
    base_type: Type
    value: Value
    dest: Value

    def resolve_uses(self, function: Function) -> None:
        self.value = self.value.resolve_uses(function)
        self.dest = self.dest.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"store {self.base_type}, {self.value}, {self.dest}"


@dataclass
class CallInstruction(Instruction):
    name: Optional[str] # name of the variable it defines
    type: Type
    function_name: str
    arguments: Tuple[Tuple[Type, Value], ...]

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        self.arguments = tuple((arg_type, value.resolve_uses(function)) for arg_type, value in self.arguments)

    def get_type(self) -> Type:
        # TODO: need to parse differently
        if isinstance(self.type, FunctionType):
            return self.type.return_type
        else:
            return self.type

    def get_full_string(self) -> str:
        prefix = "" if self.name is None else f"{self.name} = "
        return prefix + f"call {self.type} {self.function_name}({', '.join(str(value) for _, value in self.arguments)})"


@dataclass
class PhiInstruction(Instruction):
    name: str
    type: Type
    branches: OrderedDict[str, PhiBranch]

    def get_defined_variable(self) -> Optional[str]:
        return self.name

    def resolve_uses(self, function: Function) -> None:
        new_branches = OrderedDict()

        for branch in self.branches.values():
            branch.resolve_uses(function)
            new_branches[branch.label] = branch

        self.branches = new_branches

    def get_full_string(self) -> str:
        branches_string = ", ".join(f"({branch.value}, {branch.label})" for branch in self.branches.values())
        return f"{self.name} = phi {self.type}, {branches_string}"

    def get_type(self) -> Type:
        return self.type

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class PhiBranch(ASTNode):
    value: Value
    label: str

    def resolve_uses(self, function: Function) -> None:
        self.value = self.value.resolve_uses(function)

        # TODO: this is a workaround for implicitly-labeled entry block
        if self.label not in function.blocks:
            self.label = "entry"


@dataclass
class JumpInstruction(Instruction):
    label: str

    def get_full_string(self) -> str:
        return f"br label {self.label}"


@dataclass
class BranchInstruction(Instruction):
    cond: Value
    true_label: str
    false_label: str

    def resolve_uses(self, function: Function) -> None:
        self.cond = self.cond.resolve_uses(function)

    def get_full_string(self) -> str:
        return f"br {self.cond}, label {self.true_label}, label {self.false_label}"


@dataclass
class ReturnInstruction(Instruction):
    type: Type
    value: Optional[Value] = None

    def resolve_uses(self, function: Function) -> None:
        if self.value is not None:
            self.value = self.value.resolve_uses(function)

    def get_full_string(self) -> str:
        if self.value is None:
            return f"ret {self.type}"
        else:
            return f"ret {self.type}, {self.value}"
