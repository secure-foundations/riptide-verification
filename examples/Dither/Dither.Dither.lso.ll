; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @Dither(i32* noalias nocapture readonly %src, i32* nocapture %dst, i32 %rows, i32 %cols) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %cols, 0
  %smax = select i1 %0, i32 %cols, i32 0
  %1 = icmp sgt i32 %rows, 0
  %smax29 = select i1 %1, i32 %rows, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc11, %for.cond.cleanup3 ]
  %exitcond30.not = icmp eq i32 %i.0, %smax29
  br i1 %exitcond30.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  %mul = mul nsw i32 %i.0, %cols
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.body4, %for.cond1.preheader
  %err.0 = phi i32 [ %err.1, %for.body4 ], [ 0, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc, %for.body4 ], [ 0, %for.cond1.preheader ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.cond1
  %inc11 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.body4:                                        ; preds = %for.cond1
  %add = add nsw i32 %j.0, %mul
  %arrayidx = getelementptr inbounds i32, i32* %src, i32 %add
  %2 = load i32, i32* %arrayidx, align 4, !tbaa !6
  %add5 = add nsw i32 %2, %err.0
  %cmp6 = icmp sgt i32 %add5, 256
  %sub = add nsw i32 %add5, -511
  %err.1 = select i1 %cmp6, i32 %sub, i32 %add5
  %pixel.0 = select i1 %cmp6, i32 511, i32 0
  %arrayidx9 = getelementptr inbounds i32, i32* %dst, i32 %add
  store i32 %pixel.0, i32* %arrayidx9, align 4, !tbaa !6
  %inc = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10
}
