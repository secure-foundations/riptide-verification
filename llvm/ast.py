from __future__ import annotations
from typing import Tuple, Optional

from collections import OrderedDict
from dataclasses import dataclass, field


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
    ...


@dataclass
class VoidType(Type):
    def __str__(self) -> str:
        return "void"


@dataclass
class IntegerType(Type):
    bit_width: int

    def __str__(self) -> str:
        return f"i{self.bit_width}"


@dataclass
class PointerType(Type):
    base_type: Type

    def __str__(self) -> str:
        return f"{self.base_type}*"


class Value(ASTNode):
    def get_type(self) -> Type:
        raise NotImplementedError()
    
    def resolve_uses(self, function: Function) -> Value:
        return self


@dataclass
class FunctionParameter(Value):
    type: Type
    name: str

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class Function(ASTNode):
    name: str
    return_type: str

    parameters: OrderedDict[str, FunctionParameter]
    blocks: OrderedDict[str, BasicBlock]
    definitions: OrderedDict[str, Instruction] = field(init=False)

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
        for block in self.blocks.values():
            for instruction in block.instructions:
                instruction.resolve_uses(self)

    def __str__(self) -> str:
        blocks_string = "\n\n".join(str(block) for block in self.blocks.values())
        return f"{self.name}:\n{blocks_string}"


@dataclass
class BasicBlock(ASTNode):
    name: str
    instructions: Tuple[Instruction, ...]

    def __str__(self) -> str:
        instructions_string = "\n    ".join(instruction.get_full_string() for instruction in self.instructions)
        return f"  {self.name}:\n    {instructions_string}"


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
    type: Type
    value: int

    def __str__(self) -> str:
        return f"{self.type} {self.value}"


@dataclass
class NullConstant(Constant):
    base_type: Type

    def __str__(self) -> str:
        return f"{self.base_type}* null"


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

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


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
        return f"{self.name} = icmp {self.cond}, {self.type}, {self.left}, {self.right}"

    def __str__(self) -> str:
        return f"i1 {self.name}"


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

    def __str__(self) -> str:
        return f"<infer> {self.name}"


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
class PhiInstruction(Instruction):
    name: str
    type: Type
    branches: Tuple[PhiBranch, ...]

    def get_defined_variable(self) -> Optional[str]:
        return self.name
    
    def resolve_uses(self, function: Function) -> None:
        for branch in self.branches:
            branch.resolve_uses(function)

    def get_full_string(self) -> str:
        branches_string = ", ".join(f"({branch.value}, {branch.label})" for branch in self.branches)
        return f"{self.name} = phi {self.type}, {branches_string}"

    def __str__(self) -> str:
        return f"{self.type} {self.name}"


@dataclass
class PhiBranch(ASTNode):
    value: Value
    label: str

    def resolve_uses(self, function: Function) -> None:
        self.value = self.value.resolve_uses(function)


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