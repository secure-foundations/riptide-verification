; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @dmv(i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %B, i32* noalias nocapture %Z, i32 %m, i32 %n) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %n, 0
  %smax = select i1 %0, i32 %n, i32 0
  %1 = icmp sgt i32 %m, 0
  %smax23 = select i1 %1, i32 %m, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc10, %for.cond.cleanup3 ]
  %exitcond24.not = icmp eq i32 %i.0, %smax23
  br i1 %exitcond24.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  %mul = mul nsw i32 %i.0, %n
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.body4, %for.cond1.preheader
  %w.0 = phi i32 [ %add7, %for.body4 ], [ 0, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc, %for.body4 ], [ 0, %for.cond1.preheader ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.cond1
  %w.0.lcssa = phi i32 [ %w.0, %for.cond1 ]
  %arrayidx8 = getelementptr inbounds i32, i32* %Z, i32 %i.0
  store i32 %w.0.lcssa, i32* %arrayidx8, align 4, !tbaa !4
  %inc10 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8

for.body4:                                        ; preds = %for.cond1
  %add = add nsw i32 %j.0, %mul
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %add
  %2 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %arrayidx5 = getelementptr inbounds i32, i32* %B, i32 %j.0
  %3 = load i32, i32* %arrayidx5, align 4, !tbaa !4
  %mul6 = mul nsw i32 %3, %2
  %add7 = add nsw i32 %mul6, %w.0
  %inc = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10
}
