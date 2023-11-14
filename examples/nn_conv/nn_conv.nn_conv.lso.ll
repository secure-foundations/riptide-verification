; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_conv(i32* noalias nocapture readonly %weight, i32* noalias nocapture readonly %src, i32* noalias nocapture %dest, i32 %output_cols, i32 %weight_rows, i32 %weight_cols, i32 %weight_size, i32 %wc_bump, i32 %wc_wr_bump, i32 %shift) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %weight_size, 0
  %smax = select i1 %0, i32 %weight_size, i32 0
  %1 = icmp sgt i32 %output_cols, 0
  %smax49 = select i1 %1, i32 %output_cols, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc24, %for.cond.cleanup3 ]
  %exitcond50.not = icmp eq i32 %i.0, %smax49
  br i1 %exitcond50.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.inc, %for.cond1.preheader
  %w.0 = phi i32 [ %add6, %for.inc ], [ 0, %for.cond1.preheader ]
  %row.0 = phi i32 [ %row.1, %for.inc ], [ 0, %for.cond1.preheader ]
  %col.0 = phi i32 [ %col.1, %for.inc ], [ 0, %for.cond1.preheader ]
  %src_idx.0 = phi i32 [ %src_idx.1, %for.inc ], [ 0, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc15, %for.inc ], [ 0, %for.cond1.preheader ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.cond1
  %w.0.lcssa = phi i32 [ %w.0, %for.cond1 ]
  %shr = ashr i32 %w.0.lcssa, %shift
  %2 = icmp sgt i32 %shr, -32768
  %spec.store.select = select i1 %2, i32 %shr, i32 -32768
  %3 = icmp slt i32 %spec.store.select, 32767
  %spec.store.select26 = select i1 %3, i32 %spec.store.select, i32 32767
  %arrayidx22 = getelementptr inbounds i32, i32* %dest, i32 %i.0
  store i32 %spec.store.select26, i32* %arrayidx22, align 4, !tbaa !4
  %inc24 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8

for.body4:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %weight, i32 %j.0
  %4 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add = add nsw i32 %src_idx.0, %i.0
  %arrayidx5 = getelementptr inbounds i32, i32* %src, i32 %add
  %5 = load i32, i32* %arrayidx5, align 4, !tbaa !4
  %mul = mul nsw i32 %5, %4
  %add6 = add nsw i32 %mul, %w.0
  %inc = add nsw i32 %col.0, 1
  %inc7 = add nsw i32 %src_idx.0, 1
  %cmp8 = icmp eq i32 %inc, %weight_cols
  br i1 %cmp8, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body4
  %inc9 = add nsw i32 %row.0, 1
  %add10 = add nsw i32 %inc7, %wc_bump
  %cmp11 = icmp eq i32 %inc9, %weight_rows
  %spec.select = select i1 %cmp11, i32 0, i32 %inc9
  %add13 = select i1 %cmp11, i32 %wc_wr_bump, i32 0
  %spec.select48 = add nsw i32 %add10, %add13
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body4
  %row.1 = phi i32 [ %row.0, %for.body4 ], [ %spec.select, %if.then ]
  %col.1 = phi i32 [ %inc, %for.body4 ], [ 0, %if.then ]
  %src_idx.1 = phi i32 [ %inc7, %for.body4 ], [ %spec.select48, %if.then ]
  %inc15 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10
}
