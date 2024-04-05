from typing import List, Any, Union, Tuple, Optional

from collections import OrderedDict
from lark import Lark, Transformer, Token, Tree
from .ast import *


class ASTTransformer(Transformer[ASTNode]):
    # Ignore all metadata for now
    def metadata(self, args: List[Any]) -> None:
        return None

    def label(self, args: List[Token]) -> str:
        return args[0].value[1:]

    def module(self, args: List[Optional[Function]]) -> Module:
        return Module(OrderedDict((arg.name, arg) for arg in args if arg is not None))

    def function_definition(self, args: List[Any]) -> Function:
        _, return_typ, name_token, parameters, _, blocks = args
        return Function(
            name_token.value,
            return_typ,
            OrderedDict((parameter.name, parameter) for parameter in parameters),
            OrderedDict((block.name, block) for block in blocks),
        )

    def function_declaration(self, args: List[Any]) -> None:
        return None

    def basic_blocks(self, args: List[BasicBlock]) -> Tuple[BasicBlock, ...]:
        return tuple(args)

    def basic_block(self, args: List[Any]) -> BasicBlock:
        label = args[0].value[:-1]
        return BasicBlock(label, tuple(args[1:]))

    def entry_basic_block(self, args: List[BasicBlock]) -> BasicBlock:
        return args[0]

    def entry_basic_block_with_label(self, args: List[Any]) -> BasicBlock:
        label = args[0].value[:-1]
        return BasicBlock(label, tuple(args[1:]))

    def entry_basic_block_without_label(self, args: List[Any]) -> BasicBlock:
        return BasicBlock("entry", tuple(args))

    def instruction_with_metadata(self, args: List[Any]) -> Instruction:
        return args[0] # ignoring metadata

    def instruction(self, args: List[Instruction]) -> Instruction:
        return args[0]

    def add_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return AddInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def sub_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return SubInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def mul_instruction(self, args: List[Any]) -> MulInstruction:
        typ = args[1]
        return MulInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def and_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return AndInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def or_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return OrInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def xor_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return XorInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def shl_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return ShlInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def lshr_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return LshrInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def ashr_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return AshrInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def zext_instruction(self, args: List[Any]) -> AddInstruction:
        from_type = args[1]
        to_type = args[3]
        return ZextInstruction(args[0].name, from_type, to_type, args[2].attach_type(from_type))

    def sext_instruction(self, args: List[Any]) -> AddInstruction:
        from_type = args[1]
        to_type = args[3]
        return SextInstruction(args[0].name, from_type, to_type, args[2].attach_type(from_type))

    def icmp_instruction(self, args: List[Any]) -> IntegerCompareInstruction:
        typ = args[2]
        return IntegerCompareInstruction(args[0].name, args[1].value, typ, args[3].attach_type(typ), args[4].attach_type(typ))

    def select_instruction(self, args: List[Any]) -> SelectInstruction:
        typ = args[3]
        assert args[1].bit_width == 1
        return SelectInstruction(
            args[0].name,
            args[2].attach_type(IntegerType(1)),
            typ,
            args[4].attach_type(typ),
            args[6].attach_type(typ),
        )

    def getelementptr_index(self, args: List[Any]) -> Value:
        return args[1].attach_type(args[0])

    def getelementptr_instruction(self, args: List[Any]) -> GetElementPointerInstruction:
        return GetElementPointerInstruction(args[0].name, args[1], args[3].attach_type(args[2]), tuple(args[4:]))

    def load_instruction(self, args: List[Any]) -> LoadInstruction:
        return LoadInstruction(args[0].name, args[1], args[3].attach_type(args[2]))

    def store_instruction(self, args: List[Any]) -> StoreInstruction:
        return StoreInstruction(args[0], args[1].attach_type(args[0]), args[3].attach_type(args[2]))

    def call_instruction(self, args: List[CallInstruction]) -> CallInstruction:
        return args[0]

    def call_instruction_with_return(self, args: List[Any]) -> CallInstruction:
        return CallInstruction(args[0].name, args[1], args[2].name, args[3])

    def call_instruction_without_return(self, args: List[Any]) -> CallInstruction:
        return CallInstruction(None, args[0], args[1].name, args[2])

    def call_arguments(self, args: List[Tuple[Type, Value]]) -> Tuple[Tuple[Type, Value], ...]:
        return tuple((typ, value.attach_type(typ)) for typ, value in args)

    def call_argument(self, args: List[Any]) -> Tuple[Type, Value]:
        return args[0], args[1]

    def phi_instruction_branch(self, args: List[Any]) -> PhiBranch:
        return PhiBranch(args[0], args[1])

    def phi_instruction(self, args: List[Any]) -> PhiInstruction:
        return PhiInstruction(
            args[0].name,
            args[1],
            OrderedDict(
                (branch.label, PhiBranch(branch.value.attach_type(args[1]), branch.label))
                for branch in args[2:]
            ),
        )

    def jump_instruction(self, args: List[Any]) -> JumpInstruction:
        return JumpInstruction(args[0])

    def br_instruction(self, args: List[Any]) -> BranchInstruction:
        assert args[0].bit_width == 1
        return BranchInstruction(args[1].attach_type(IntegerType(1)), args[2], args[3])

    def ret_instruction(self, args: List[Any]) -> ReturnInstruction:
        return ReturnInstruction(args[0], args[1].attach_type(args[0]) if len(args) == 2 else None)

    def parameters(self, args: List[FunctionParameter]) -> Tuple[FunctionParameter, ...]:
        return tuple(args)

    def parameter(self, args: List[Any]) -> List[FunctionParameter]:
        return FunctionParameter(args[0], args[2].value, args[1])

    def parameter_attributes(self, args: List[Any]) -> Tuple[str, ...]:
        return tuple(arg for arg in args if arg is not None)

    def parameter_attribute_ignore(self, args: List[Any]) -> None:
        return None

    def parameter_attribute_noalias(self, args: List[Any]) -> str:
        return "noalias"

    def type(self, args: List[Type]) -> Type:
        return args[0]

    def void_type(self, args: List[Token]) -> VoidType:
        return VoidType()

    def integer_type(self, args: List[Token]) -> IntegerType:
        return IntegerType(int(args[0].value[1:]))

    def pointer_type(self, args: List[Type]) -> PointerType:
        return PointerType(args[0])

    def array_type(self, args: List[Type]) -> ArrayType:
        return ArrayType(args[1], int(args[0].value))

    def call_return_type(self, args: List[FunctionType]) -> FunctionType:
        return args[0]

    def function_type_with_eclipsis(self, args: List[Type]) -> FunctionType:
        return FunctionType(args[0], tuple(args[1:]), True)

    def function_type_without_eclipsis(self, args: List[Type]) -> FunctionType:
        return FunctionType(args[0], tuple(args[1:]), False)

    def value(self, args: List[UnresolvedValue]) -> UnresolvedValue:
        return args[0]

    def variable(self, args: List[Token]) -> UnresolvedVariable:
        return UnresolvedVariable(args[0].value)

    def integer_value(self, args: List[Token]) -> UnresolvedIntegerValue:
        return UnresolvedIntegerValue(int(args[0].value))

    def true_value(self, args: List[Token]) -> UnresolvedIntegerValue:
        return UnresolvedIntegerValue(int(1))

    def false_value(self, args: List[Token]) -> UnresolvedIntegerValue:
        return UnresolvedIntegerValue(int(0))

    def null_value(self, args: List[Token]) -> UnresolvedNullValue:
        return UnresolvedNullValue()

    def undef_value(self, args: List[Token]) -> UnresolvedUndefValue:
        return UnresolvedUndefValue()


class Parser:
    SYNTAX = r"""
        %import common.ESCAPED_STRING -> STRING

        INLINE_COMMENT: /;[^\n]*/

        %ignore INLINE_COMMENT
        %ignore /[ \n\t\f\r]+/

        INTEGER: /-?(0|[1-9][0-9]*)/
        INTEGER_TYPE: /i[1-9][0-9]*/
        VARIABLE: /(%|@)((0|[1-9][0-9]*)|([A-Za-z._][A-Za-z0-9-_'.]*))/
        BLOCK_LABEL: /[A-Za-z0-9.][A-Za-z0-9-_'.]*:/
        METADATA_LABEL: /!((0|[1-9][0-9]*)|([A-Za-z][A-Za-z0-9-_'.]*))/
        ATTRIBUTE_GROUP_NAME: /\#((0|[1-9][0-9]*)|([A-Za-z][A-Za-z0-9-_'.]*))/
        IDENTIFIER.0: /[A-Za-z][A-Za-z0-9-_'.]*/

        module: (metadata|function_definition|function_declaration)*

        type: "void" -> void_type
            | integer_type
            | pointer_type
            | array_type

        call_return_type: type "(" [type ("," type)*] "," "..." ")" -> function_type_with_eclipsis
                        | type "(" [type ("," type)*] ")" -> function_type_without_eclipsis
                        | type

        integer_type: INTEGER_TYPE
        pointer_type: type "*"
        array_type: "[" INTEGER "x" type "]"

        function_definition: "define" function_attributes type VARIABLE "(" parameters ")" function_attributes "{" basic_blocks "}"

        function_declaration: "declare" function_attributes type VARIABLE "(" [type ("," type)*] ")" function_attributes

        parameters: [parameter ("," parameter)*]
        parameter: type parameter_attributes VARIABLE

        basic_blocks: entry_basic_block basic_block*
        basic_block: BLOCK_LABEL instruction_with_metadata*
        entry_basic_block: instruction_with_metadata* -> entry_basic_block_without_label
                         | BLOCK_LABEL instruction_with_metadata* -> entry_basic_block_with_label

        instruction_with_metadata: instruction ("," METADATA_LABEL+)*

        instruction: add_instruction
                   | sub_instruction
                   | mul_instruction
                   | and_instruction
                   | or_instruction
                   | xor_instruction
                   | shl_instruction
                   | lshr_instruction
                   | ashr_instruction
                   | zext_instruction
                   | sext_instruction
                   | icmp_instruction
                   | select_instruction
                   | getelementptr_instruction
                   | load_instruction
                   | store_instruction
                   | call_instruction
                   | phi_instruction
                   | jump_instruction
                   | br_instruction
                   | ret_instruction

        add_instruction: variable "=" "add" ["nuw"] ["nsw"] integer_type value "," value
        sub_instruction: variable "=" "sub" ["nuw"] ["nsw"] integer_type value "," value
        mul_instruction: variable "=" "mul" ["nuw"] ["nsw"] integer_type value "," value

        and_instruction: variable "=" "and" integer_type value "," value
        or_instruction: variable "=" "or" integer_type value "," value
        xor_instruction: variable "=" "xor" integer_type value "," value
        shl_instruction: variable "=" "shl" ["nuw"] ["nsw"] integer_type value "," value
        lshr_instruction: variable "=" "lshr" ["exact"] integer_type value "," value
        ashr_instruction: variable "=" "ashr" ["exact"] integer_type value "," value

        zext_instruction: variable "=" "zext" integer_type value "to" integer_type
        sext_instruction: variable "=" "sext" integer_type value "to" integer_type

        icmp_instruction: variable "=" "icmp" ICMP_CONDITION type value "," value
        ICMP_CONDITION.1: /eq|ne|ugt|uge|ult|ule|sgt|sge|slt|sle/

        select_instruction: variable "=" "select" integer_type value "," type value "," type value

        getelementptr_instruction: variable "=" "getelementptr" ["inbounds"] type "," pointer_type value ("," ["inrange"] getelementptr_index)*
        getelementptr_index: integer_type value

        load_instruction: variable "=" "load" type "," pointer_type value ["," "align" INTEGER]

        store_instruction: "store" type value "," pointer_type value ["," "align" INTEGER]

        call_instruction: variable "=" ["tail"] "call" call_return_type variable "(" call_arguments ")" -> call_instruction_with_return
                        | ["tail"] "call" call_return_type variable "(" call_arguments ")" -> call_instruction_without_return
        call_arguments: [call_argument ("," call_argument)*]
        call_argument: type value

        phi_instruction: variable "=" "phi" type phi_instruction_branch ("," phi_instruction_branch)*
        phi_instruction_branch: "[" value "," label "]"

        jump_instruction: "br" "label" label

        br_instruction: "br" integer_type value "," "label" label "," "label" label

        ret_instruction: "ret" type [value]

        label: VARIABLE
        variable: VARIABLE

        value: INTEGER -> integer_value
             | "null" -> null_value
             | variable
             | "undef" -> undef_value
             | "true" -> true_value
             | "false" -> false_value

        /////////////////////////////////////////////////////
        // The portion of the syntax below will be ignored //
        // in our AST.                                     //
        /////////////////////////////////////////////////////

        function_attributes: function_attribute*

        function_attribute: ATTRIBUTE_GROUP_NAME
                          | "dso_local"
                          | "local_unnamed_addr"
                          | "arm_aapcscc"

        parameter_attributes: parameter_attribute*

        parameter_attribute: "zeroext" -> parameter_attribute_ignore
                           | "signext" -> parameter_attribute_ignore
                           | "inreg" -> parameter_attribute_ignore
                           | "noalias" -> parameter_attribute_noalias
                           | "nocapture" -> parameter_attribute_ignore
                           | "readonly" -> parameter_attribute_ignore
                           | "readnone" -> parameter_attribute_ignore

        metadata: "source_filename" "=" STRING
                | "target" "datalayout" "=" STRING
                | "target" "triple" "=" STRING
                | METADATA_LABEL "=" ["distinct"] metadata_value
                | "attributes" ATTRIBUTE_GROUP_NAME "=" "{" attribute_key_value* "}"

        attribute_key_value: IDENTIFIER
                           | STRING
                           | parameter_attribute
                           | IDENTIFIER "=" attribute_value
                           | STRING "=" attribute_value

        attribute_value: STRING
                       | IDENTIFIER
                       | INTEGER

        metadata_value: INTEGER_TYPE INTEGER
                      | METADATA_LABEL
                      | array_type "[" [metadata_value ["," metadata_value]*] "]"
                      | "!" STRING
                      | "!" "{" [metadata_value ("," metadata_value)*] "}"
    """

    MODULE_PARSER = Lark(
        SYNTAX,
        start="module",
        parser="lalr",
        lexer="standard",
        propagate_positions=True,
    )

    @staticmethod
    def parse_module(src: str) -> Module:
        raw_ast = Parser.MODULE_PARSER.parse(src)
        ast = ASTTransformer().transform(raw_ast)
        ast.resolve_uses()
        return ast
