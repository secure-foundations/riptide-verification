from __future__ import annotations
from typing import Tuple, Optional

from collections import OrderedDict
from dataclasses import dataclass, field


class ASTNode:
    ...


@dataclass
class Module(ASTNode):
    functions: Tuple[Function, ...]


class Type(ASTNode):
    ...


@dataclass
class VoidType(Type):
    ...


@dataclass
class IntegerType(Type):
    bit_width: int


@dataclass
class PointerType(Type):
    base_type: Type


@dataclass
class FunctionParameter(ASTNode):
    type: Type
    name: str


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


@dataclass
class BasicBlock(ASTNode):
    name: str
    instructions: Tuple[Instruction, ...]


class Value(ASTNode):
    def get_type(self) -> Type:
        raise NotImplementedError()


# Values who types cannot be inferred yet
class UnresolvedValue(Value):
    def attach_type(self, typ: Type) -> Value:
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


class Constant(Value):
    ...


@dataclass
class IntegerConstant(Constant):
    type: Type
    value: int


@dataclass
class NullConstant(Constant):
    base_type: Type


class Instruction(Value):
    def get_defined_variable(self) -> Optional[str]:
        return None


@dataclass
class AddInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name


@dataclass
class MulInstruction(Instruction):
    name: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name


@dataclass
class IntegerCompareInstruction(Instruction):
    name: str
    cond: str
    type: Type
    left: Value
    right: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name


@dataclass
class GetElementPointerInstruction(Instruction):
    name: str
    base_type: Type
    pointer: Value
    indices: Tuple[Value, ...]

    def get_defined_variable(self) -> Optional[str]:
        return self.name


@dataclass
class LoadInstruction(Instruction):
    name: str
    base_type: Type
    pointer: Value

    def get_defined_variable(self) -> Optional[str]:
        return self.name


@dataclass
class StoreInstruction(Instruction):
    base_type: Type
    value: Value
    dest: Value


@dataclass
class PhiInstruction(Instruction):
    name: str
    type: Type
    branches: Tuple[PhiBranch, ...]

    def get_defined_variable(self) -> Optional[str]:
        return self.name


@dataclass
class PhiBranch(ASTNode):
    value: Value
    label: str


@dataclass
class JumpInstruction(Instruction):
    label: str


@dataclass
class BranchInstruction(Instruction):
    cond: Value
    true_label: str
    false_label: str


@dataclass
class ReturnInstruction(Instruction):
    type: Type
    value: Optional[Value] = None
