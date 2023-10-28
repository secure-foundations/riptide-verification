; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @test(i32* nocapture %A, i32* nocapture readonly %B, i32 %lenA, i32 %lenB) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %lenB, 0
  %smax = select i1 %0, i32 %lenB, i32 0
  %1 = icmp sgt i32 %lenA, 0
  %smax15 = select i1 %1, i32 %lenA, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %lso.alloc1.1.lcssa, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc7, %for.cond.cleanup3 ]
  %exitcond16.not = icmp eq i32 %i.0, %smax15
  br i1 %exitcond16.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i32 %i.0
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.body4, %for.cond1.preheader
  %lso.alloc1.1 = phi i32 [ %lso.alloc1.0, %for.cond1.preheader ], [ %3, %for.body4 ]
  %j.0 = phi i32 [ %inc, %for.body4 ], [ 0, %for.cond1.preheader ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.cond1
  %lso.alloc1.1.lcssa = phi i32 [ %lso.alloc1.1, %for.cond1 ]
  %inc7 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.body4:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %j.0
  %2 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx, i32 %lso.alloc1.1)
  %3 = call i32 (i32, i32*, ...) @cgra_store32(i32 %2, i32* %arrayidx5)
  %inc = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !6
}
