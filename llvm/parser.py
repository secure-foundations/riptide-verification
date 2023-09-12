from lark import Lark, Transformer, Token, Tree

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

        type: "void"
            | INTEGER_TYPE
            | type "*"

        function: "define" function_attribute* type VARIABLE "(" [parameter ("," parameter)*] ")" function_attribute* "{" basic_block* "}"

        basic_block: BLOCK_LABEL instruction*

        register: VARIABLE

        value: INTEGER_TYPE INTEGER
             | type "*" "null"
             | register

        instruction: add_instruction
                   | ret_instruction

        add_instruction: register "=" "add" type value "," value

        ret_instruction: "ret" type [value]

        parameter: type parameter_attribute* VARIABLE

        function_attribute: ATTRIBUTE_GROUP_NAME
                          | "dso_local"
                          | "local_unnamed_addr"

        parameter_attribute: "noalias"
                           | "nocapture"
                           | "readonly"

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
    def parse_module(src: str):
        print(Parser.MODULE_PARSER.parse(src))


if __name__ == "__main__":
    Parser.parse_module(r"""
; ModuleID = 'dmv.c'
source_filename = "dmv.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx12.0.0"

; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @dmv(i16* noalias nocapture readonly %A, i16* noalias nocapture readonly %B, i16* noalias nocapture %Z, i32 %m, i32 %n) local_unnamed_addr #0 {
entry:
  ret void
}

attributes #0 = { minsize nofree norecurse nounwind optsize ssp "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-bf16,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }
                        
!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 12.0.0 (https://github.com/sgh185/LLVM_installer.git 65fdfddc0fbfd3889dd84137547b068ce7c48ff8)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}

""")

