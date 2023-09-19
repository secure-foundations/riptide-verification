from .parser import Parser
from .state import Configuration, NextConfiguration, FunctionReturn

if __name__ == "__main__":
    module = Parser.parse_module(r"""
; ModuleID = 'test-1.base.ll'
source_filename = "test-1.c"
target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv4t-apple-macosx12.0.0"

; Function Attrs: minsize nofree norecurse nounwind optsize ssp writeonly
define dso_local void @test(i32* noalias nocapture %array, i32 %len) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %len, 0
  %smax = select i1 %0, i32 %len, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %add = add nuw nsw i32 %i.0, %sum.0
  ; %arrayidx = getelementptr inbounds i32, i32* %array, i32 %i.0
  store i32 %add, i32* %array, align 4, !tbaa !4
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

    config = Configuration.get_initial_configuration(module, module.functions["@test"])

    print(config)

    queue = [config]
    final_state_count = 0

    while len(queue) != 0:
        config = queue.pop(0)
        results = config.step()
        for result in results:
            if isinstance(result, NextConfiguration):
                config = result.config
                queue.append(config)
                # print(config)
            elif isinstance(result, FunctionReturn):
                final_state_count += 1
                print(f"\nfinal config #{final_state_count}")
                print(f"returns {result.value}, {result.final_config}")
            else:
                assert False, f"unsupported result {result}"

    # results = config.step()
    # for result in results:
    #     if isinstance(result, NextConfiguration):
    #         config = result.config
    #         print(config)
