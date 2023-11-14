; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_vadd(i32* noalias nocapture readonly %weight, i32* noalias nocapture readonly %src, i32* noalias nocapture %dest, i32 %size) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %size, 0
  %smax = select i1 %0, i32 %size, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %src, i32 %i.0
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %arrayidx1 = getelementptr inbounds i32, i32* %weight, i32 %i.0
  %2 = load i32, i32* %arrayidx1, align 4, !tbaa !4
  %add = add nsw i32 %2, %1
  %arrayidx2 = getelementptr inbounds i32, i32* %dest, i32 %i.0
  store i32 %add, i32* %arrayidx2, align 4, !tbaa !4
  %inc = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8
}
