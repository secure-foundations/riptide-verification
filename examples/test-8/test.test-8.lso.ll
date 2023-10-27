; Function Attrs: minsize nofree norecurse nounwind optsize ssp writeonly
define dso_local void @test(i32* nocapture %A, i32* nocapture %B, i32 %len) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %len, 0
  %smax17 = select i1 %0, i32 %len, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %lso.alloc.0 = phi i32 [ 0, %entry ], [ %1, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %exitcond18.not = icmp eq i32 %i.0, %smax17
  br i1 %exitcond18.not, label %for.cond2.preheader, label %for.body

for.cond2.preheader:                              ; preds = %for.cond
  %lso.alloc.0.lcssa = phi i32 [ %lso.alloc.0, %for.cond ]
  br label %for.cond2

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.0
  %1 = call i32 (i32, i32*, ...) @cgra_store32(i32 0, i32* %arrayidx)
  %inc = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.cond2:                                        ; preds = %for.body5, %for.cond2.preheader
  %i1.0 = phi i32 [ %inc8, %for.body5 ], [ 0, %for.cond2.preheader ]
  %exitcond.not = icmp eq i32 %i1.0, %smax17
  br i1 %exitcond.not, label %for.cond.cleanup4, label %for.body5

for.cond.cleanup4:                                ; preds = %for.cond2
  ret void

for.body5:                                        ; preds = %for.cond2
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i32 %i1.0
  %2 = call i32 (i32, i32*, ...) @cgra_store32(i32 1, i32* %arrayidx6, i32 %lso.alloc.0.lcssa)
  %inc8 = add nuw i32 %i1.0, 1
  br label %for.cond2, !llvm.loop !6
}
