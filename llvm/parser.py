from typing import List, Any, Union, Tuple, Optional

from collections import OrderedDict
from lark import Lark, Transformer, Token, Tree
from .ast import (
    ASTNode, Module, Function,
    BasicBlock,
    Type, VoidType, IntegerType, PointerType, ArrayType, FunctionParameter,
    UnresolvedValue, UnresolvedIntegerValue, UnresolvedNullValue, UnresolvedVariable,
    Value,
    Instruction, AddInstruction, MulInstruction, IntegerCompareInstruction, SelectInstruction,
    GetElementPointerInstruction, LoadInstruction, StoreInstruction,
    PhiInstruction, PhiBranch,
    JumpInstruction, BranchInstruction, ReturnInstruction,
)


class ASTTransformer(Transformer[ASTNode]):
    # Ignore all metadata for now
    def metadata(self, args: List[Any]) -> None:
        return None

    def label(self, args: List[Token]) -> str:
        return args[0].value[1:]

    def module(self, args: List[Optional[Function]]) -> Module:
        return Module(OrderedDict((arg.name, arg) for arg in args if arg is not None))

    def function(self, args: List[Any]) -> Function:
        _, return_typ, name_token, parameters, _, blocks = args
        return Function(
            name_token.value,
            return_typ,
            OrderedDict((parameter.name, parameter) for parameter in parameters),
            OrderedDict((block.name, block) for block in blocks),
        )

    def basic_blocks(self, args: List[BasicBlock]) -> Tuple[BasicBlock, ...]:
        return tuple(args)

    def basic_block(self, args: List[Any]) -> BasicBlock:
        label = args[0].value[:-1]
        return BasicBlock(label, tuple(args[1:]))

    def instruction_with_metadata(self, args: List[Any]) -> Instruction:
        return args[0] # ignoring metadata

    def instruction(self, args: List[Instruction]) -> Instruction:
        return args[0]
    
    def add_instruction(self, args: List[Any]) -> AddInstruction:
        typ = args[1]
        return AddInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))
    
    def mul_instruction(self, args: List[Any]) -> MulInstruction:
        typ = args[1]
        return MulInstruction(args[0].name, typ, args[2].attach_type(typ), args[3].attach_type(typ))

    def icmp_instruction(self, args: List[Any]) -> IntegerCompareInstruction:
        typ = args[2]
        return IntegerCompareInstruction(args[0].name, args[1].value, typ, args[3].attach_type(typ), args[4].attach_type(typ))

    def select_instruction(self, args: List[Any]) -> SelectInstruction:
        typ = args[2]
        return SelectInstruction(
            args[0].name,
            args[1].attach_type(IntegerType(1)),
            typ,
            args[3].attach_type(typ),
            args[5].attach_type(typ),
        )

    def getelementptr_index(self, args: List[Any]) -> Value:
        return args[1].attach_type(args[0])

    def getelementptr_instruction(self, args: List[Any]) -> GetElementPointerInstruction:
        return GetElementPointerInstruction(args[0].name, args[1], args[3].attach_type(args[2]), tuple(args[4:]))

    def load_instruction(self, args: List[Any]) -> LoadInstruction:
        return LoadInstruction(args[0].name, args[1], args[3].attach_type(args[2]))
    
    def store_instruction(self, args: List[Any]) -> StoreInstruction:
        return StoreInstruction(args[0], args[1].attach_type(args[0]), args[3].attach_type(args[2]))

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
        return BranchInstruction(args[0].attach_type(IntegerType(1)), args[1], args[2])
    
    def ret_instruction(self, args: List[Any]) -> ReturnInstruction:
        return ReturnInstruction(args[0], args[1].attach_type(args[0]) if len(args) == 2 else None)

    def parameters(self, args: List[FunctionParameter]) -> Tuple[FunctionParameter, ...]:
        return tuple(args)

    def parameter(self, args: List[Any]) -> List[FunctionParameter]:
        return FunctionParameter(args[0], args[2].value)

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

    def value(self, args: List[UnresolvedValue]) -> UnresolvedValue:
        return args[0]
    
    def variable(self, args: List[Token]) -> UnresolvedVariable:
        return UnresolvedVariable(args[0].value)

    def integer_value(self, args: List[Token]) -> UnresolvedIntegerValue:
        return UnresolvedIntegerValue(int(args[0].value))
    
    def null_value(self, args: List[Token]) -> UnresolvedNullValue:
        return UnresolvedNullValue()


class Parser:
    SYNTAX = r"""
        %import common.ESCAPED_STRING -> STRING

        INLINE_COMMENT: /;[^\n]*/

        %ignore INLINE_COMMENT
        %ignore /[ \n\t\f\r]+/

        INTEGER: /0|[1-9][0-9]*/
        INTEGER_TYPE: /i[1-9][0-9]*/
        VARIABLE: /(%|@)((0|[1-9][0-9]*)|([A-Za-z][A-Za-z0-9-_'.]*))/
        BLOCK_LABEL: /[A-Za-z][A-Za-z0-9-_'.]*:/
        METADATA_LABEL: /!((0|[1-9][0-9]*)|([A-Za-z][A-Za-z0-9-_'.]*))/
        ATTRIBUTE_GROUP_NAME: /\#((0|[1-9][0-9]*)|([A-Za-z][A-Za-z0-9-_'.]*))/
        IDENTIFIER.0: /[A-Za-z][A-Za-z0-9-_'.]*/

        module: (metadata|function)*

        type: "void" -> void_type
            | integer_type
            | pointer_type
            | array_type

        integer_type: INTEGER_TYPE
        pointer_type: type "*"
        array_type: "[" INTEGER "x" type "]"
        
        function: "define" function_attributes type VARIABLE "(" parameters ")" function_attributes "{" basic_blocks "}"
        
        parameters: [parameter ("," parameter)*]
        parameter: type parameter_attributes VARIABLE
        
        basic_blocks: basic_block*
        basic_block: BLOCK_LABEL instruction_with_metadata*

        instruction_with_metadata: instruction ("," METADATA_LABEL+)*
        
        instruction: add_instruction
                   | mul_instruction
                   | icmp_instruction
                   | select_instruction
                   | getelementptr_instruction
                   | load_instruction
                   | store_instruction
                   | phi_instruction
                   | jump_instruction
                   | br_instruction
                   | ret_instruction

        add_instruction: variable "=" "add" ["nuw"] ["nsw"] integer_type value "," value
        mul_instruction: variable "=" "mul" ["nuw"] ["nsw"] integer_type value "," value

        icmp_instruction: variable "=" "icmp" ICMP_CONDITION type value "," value
        ICMP_CONDITION.1: /eq|ne|ugt|uge|ult|ule|sgt|sge|slt|sle/

        select_instruction: variable "=" "select" "i1" value "," type value "," type value
 
        getelementptr_instruction: variable "=" "getelementptr" ["inbounds"] type "," pointer_type value ("," ["inrange"] getelementptr_index)*
        getelementptr_index: integer_type value

        load_instruction: variable "=" "load" type "," pointer_type value ["," "align" INTEGER]

        store_instruction: "store" type value "," pointer_type value ["," "align" INTEGER]

        phi_instruction: variable "=" "phi" type phi_instruction_branch ("," phi_instruction_branch)*
        phi_instruction_branch: "[" value "," label "]"
        
        jump_instruction: "br" "label" label

        br_instruction: "br" "i1" value "," "label" label "," "label" label
        
        ret_instruction: "ret" type [value]

        label: VARIABLE
        variable: VARIABLE

        value: INTEGER -> integer_value
             | "null" -> null_value
             | variable

        /////////////////////////////////////////////////////
        // The portion of the syntax below will be ignored //
        // in our AST.                                     //
        /////////////////////////////////////////////////////

        function_attributes: function_attribute*

        function_attribute: ATTRIBUTE_GROUP_NAME
                          | "dso_local"
                          | "local_unnamed_addr"

        parameter_attributes: parameter_attribute*
                          
        parameter_attribute: "zeroext"
                           | "signext"
                           | "inreg"
                           | "noalias"
                           | "nocapture"
                           | "readonly"
                           | "readnone"

        metadata: "source_filename" "=" STRING
                | "target" "datalayout" "=" STRING
                | "target" "triple" "=" STRING
                | METADATA_LABEL "=" ["distinct"] metadata_value
                | "attributes" ATTRIBUTE_GROUP_NAME "=" "{" attribute_key_value* "}"

        attribute_key_value: IDENTIFIER
                           | STRING
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


if __name__ == "__main__":
    module = Parser.parse_module(r"""
; ModuleID = 'parallel.base.ll'
source_filename = "parallel.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx12.0.0"

; Function Attrs: minsize nofree norecurse nounwind optsize ssp writeonly
define dso_local void @f(i32* noalias nocapture %array1, i32* noalias nocapture %array2, i32 %len) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %len, 0
  %smax = select i1 %0, i32 %len, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %array1, i32 %i.0
  store i32 %i.0, i32* %arrayidx, align 4, !tbaa !4
  %arrayidx1 = getelementptr inbounds i32, i32* %array2, i32 %i.0
  store i32 %len, i32* %arrayidx1, align 4, !tbaa !4
  %inc = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8
}

attributes #0 = { minsize nofree norecurse nounwind optsize ssp writeonly "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-bf16,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 12.0.0 (https://github.com/sgh185/LLVM_installer.git 65fdfddc0fbfd3889dd84137547b068ce7c48ff8)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}

""")
    print(module)

