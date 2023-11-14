; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_fc(i32* noalias nocapture readonly %weight, i32* noalias nocapture readonly %src, i32* noalias nocapture %dest, i32 %rows, i32 %cols, i32 %shift) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %cols, 0
  %smax = select i1 %0, i32 %cols, i32 0
  %1 = icmp sgt i32 %rows, 0
  %smax33 = select i1 %1, i32 %rows, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc14, %for.cond.cleanup3 ]
  %exitcond34.not = icmp eq i32 %i.0, %smax33
  br i1 %exitcond34.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  %mul = mul nsw i32 %i.0, %cols
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
  %shr = ashr i32 %w.0.lcssa, %shift
  %2 = icmp sgt i32 %shr, -32768
  %spec.store.select = select i1 %2, i32 %shr, i32 -32768
  %3 = icmp slt i32 %spec.store.select, 32767
  %spec.store.select16 = select i1 %3, i32 %spec.store.select, i32 32767
  %arrayidx12 = getelementptr inbounds i32, i32* %dest, i32 %i.0
  store i32 %spec.store.select16, i32* %arrayidx12, align 4, !tbaa !4
  %inc14 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8

for.body4:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %src, i32 %j.0
  %4 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add = add nsw i32 %j.0, %mul
  %arrayidx5 = getelementptr inbounds i32, i32* %weight, i32 %add
  %5 = load i32, i32* %arrayidx5, align 4, !tbaa !4
  %mul6 = mul nsw i32 %5, %4
  %add7 = add nsw i32 %mul6, %w.0
  %inc = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10
}
