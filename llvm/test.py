from .parser import Parser
from .state import Configuration, NextConfiguration

if __name__ == "__main__":
    module = Parser.parse_module(r"""
; ModuleID = 'branch-2.c'
source_filename = "branch-2.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx12.0.0"

; Function Attrs: minsize nofree norecurse nounwind optsize ssp willreturn writeonly
define dso_local void @foo(i32 %i, i32* noalias nocapture %a, i32* noalias nocapture %b) local_unnamed_addr #0 {
entry:
  %cmp = icmp eq i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, i32* %a, align 4, !tbaa !4
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, i32* %b, align 4, !tbaa !4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

attributes #0 = { minsize nofree norecurse nounwind optsize ssp willreturn writeonly "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+armv4t,+soft-float,+strict-align,-bf16,-crypto,-d32,-dotprod,-fp-armv8,-fp-armv8d16,-fp-armv8d16sp,-fp-armv8sp,-fp16,-fp16fml,-fp64,-fpregs,-fullfp16,-mve,-mve.fp,-neon,-thumb-mode,-vfp2,-vfp2sp,-vfp3,-vfp3d16,-vfp3d16sp,-vfp3sp,-vfp4,-vfp4d16,-vfp4d16sp,-vfp4sp" "unsafe-fp-math"="false" "use-soft-float"="true" }

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
""")
    print(module)

    config = Configuration.get_initial_configuration(module, module.functions["@foo"])

    print(config)

    # queue = [config]

    # while len(queue) != 0:
    #     config = queue.pop()
    #     results = config.step()
    #     for result in results:
    #         if isinstance(result, NextConfiguration):
    #             config = result.config
    #             queue

    results = config.step()
    for result in results:
        if isinstance(result, NextConfiguration):
            config = result.config
            print(config)
