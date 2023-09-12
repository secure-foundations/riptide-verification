from __future__ import annotations

from collections import OrderedDict
from typing import Tuple
from dataclasses import dataclass


class ASTNode:
    ...


@dataclass
class Module(ASTNode):
    ...


class Type(ASTNode):
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
    parameters: Tuple[FunctionParameter, ...]
    blocks: OrderedDict[str, BasicBlock]


@dataclass
class BasicBlock(ASTNode):
    name: str
    instructions: Tuple[Instruction, ...]


class Value(ASTNode):
    def get_type(self) -> Type:
        raise NotImplementedError()


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
    ...


@dataclass
class AddInstruction(Instruction):
    type: Type
    left: Value
    right: Value
