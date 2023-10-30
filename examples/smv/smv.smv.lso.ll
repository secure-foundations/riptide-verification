; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @smv(i32* noalias nocapture readonly %Arow, i32* noalias nocapture readonly %Acol, i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %B, i32* noalias nocapture %Z, i32 %m) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %m, 0
  %smax = select i1 %0, i32 %m, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup4, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup4 ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %Arow, i32 %i.0
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add = add nuw i32 %i.0, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %Arow, i32 %add
  %2 = load i32, i32* %arrayidx1, align 4, !tbaa !4
  br label %for.cond2

for.cond2:                                        ; preds = %for.body5, %for.body
  %w.0 = phi i32 [ 0, %for.body ], [ %add9, %for.body5 ]
  %j.0 = phi i32 [ %1, %for.body ], [ %inc, %for.body5 ]
  %cmp3 = icmp slt i32 %j.0, %2
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %for.cond2
  %w.0.lcssa = phi i32 [ %w.0, %for.cond2 ]
  %arrayidx10 = getelementptr inbounds i32, i32* %Z, i32 %i.0
  store i32 %w.0.lcssa, i32* %arrayidx10, align 4, !tbaa !4
  br label %for.cond, !llvm.loop !8

for.body5:                                        ; preds = %for.cond2
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %j.0
  %3 = load i32, i32* %arrayidx6, align 4, !tbaa !4
  %arrayidx7 = getelementptr inbounds i32, i32* %Acol, i32 %j.0
  %4 = load i32, i32* %arrayidx7, align 4, !tbaa !4
  %arrayidx8 = getelementptr inbounds i32, i32* %B, i32 %4
  %5 = load i32, i32* %arrayidx8, align 4, !tbaa !4
  %mul = mul nsw i32 %5, %3
  %add9 = add nsw i32 %mul, %w.0
  %inc = add nsw i32 %j.0, 1
  br label %for.cond2, !llvm.loop !10
}
