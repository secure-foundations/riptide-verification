; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_conv(i32* nocapture readonly %weight, i32* nocapture readonly %src, i32* nocapture %dest, i32 %output_cols, i32 %weight_rows, i32 %weight_cols, i32 %weight_size, i32 %wc_bump, i32 %wc_wr_bump, i32 %shift) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %weight_size, 0
  %smax = select i1 %0, i32 %weight_size, i32 0
  %1 = icmp sgt i32 %output_cols, 0
  %smax52 = select i1 %1, i32 %output_cols, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %lso.alloc.0 = phi i32 [ 0, %entry ], [ %4, %for.cond.cleanup3 ]
  %lso.alloc2.0 = phi i32 [ 0, %entry ], [ %4, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc23, %for.cond.cleanup3 ]
  %src_ptr.0 = phi i32* [ %src, %entry ], [ %incdec.ptr, %for.cond.cleanup3 ]
  %exitcond53.not = icmp eq i32 %i.0, %smax52
  br i1 %exitcond53.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.inc, %for.cond1.preheader
  %w.0 = phi i32 [ %add, %for.inc ], [ 0, %for.cond1.preheader ]
  %row.0 = phi i32 [ %row.1, %for.inc ], [ 0, %for.cond1.preheader ]
  %col.0 = phi i32 [ %col.1, %for.inc ], [ 0, %for.cond1.preheader ]
  %src_idx.0 = phi i32 [ %src_idx.1, %for.inc ], [ 0, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc14, %for.inc ], [ 0, %for.cond1.preheader ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.cond1
  %w.0.lcssa = phi i32 [ %w.0, %for.cond1 ]
  %shr = ashr i32 %w.0.lcssa, %shift
  %2 = icmp sgt i32 %shr, -32768
  %spec.store.select = select i1 %2, i32 %shr, i32 -32768
  %3 = icmp slt i32 %spec.store.select, 32767
  %spec.store.select25 = select i1 %3, i32 %spec.store.select, i32 32767
  %arrayidx21 = getelementptr inbounds i32, i32* %dest, i32 %i.0
  %4 = call i32 (i32, i32*, ...) @cgra_store32(i32 %spec.store.select25, i32* %arrayidx21)
  %incdec.ptr = getelementptr inbounds i32, i32* %src_ptr.0, i32 1
  %inc23 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.body4:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %weight, i32 %j.0
  %5 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx, i32 %lso.alloc2.0)
  %arrayidx5 = getelementptr inbounds i32, i32* %src_ptr.0, i32 %src_idx.0
  %6 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx5, i32 %lso.alloc.0)
  %mul = mul nsw i32 %6, %5
  %add = add nsw i32 %mul, %w.0
  %inc = add nsw i32 %col.0, 1
  %inc6 = add nsw i32 %src_idx.0, 1
  %cmp7 = icmp eq i32 %inc, %weight_cols
  br i1 %cmp7, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body4
  %inc8 = add nsw i32 %row.0, 1
  %add9 = add nsw i32 %inc6, %wc_bump
  %cmp10 = icmp eq i32 %inc8, %weight_rows
  %spec.select = select i1 %cmp10, i32 0, i32 %inc8
  %add12 = select i1 %cmp10, i32 %wc_wr_bump, i32 0
  %spec.select51 = add nsw i32 %add9, %add12
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body4
  %row.1 = phi i32 [ %row.0, %for.body4 ], [ %spec.select, %if.then ]
  %col.1 = phi i32 [ %inc, %for.body4 ], [ 0, %if.then ]
  %src_idx.1 = phi i32 [ %inc6, %for.body4 ], [ %spec.select51, %if.then ]
  %inc14 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !6
}
