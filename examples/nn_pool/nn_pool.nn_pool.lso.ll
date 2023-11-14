; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_pool(i32* nocapture readonly %src, i32* nocapture %dest, i32 %input_rows_bump, i32 %input_cols, i32 %output_size, i32 %output_cols, i32 %pool_size) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %pool_size, 0
  %smax = select i1 %0, i32 %pool_size, i32 0
  %1 = icmp sgt i32 %output_size, 0
  %smax48 = select i1 %1, i32 %output_size, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %lso.alloc.0 = phi i32 [ 0, %entry ], [ %2, %for.cond.cleanup3 ]
  %col.0 = phi i32 [ 0, %entry ], [ %spec.select, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc21, %for.cond.cleanup3 ]
  %src_ptr.0 = phi i32* [ %src, %entry ], [ %spec.select45, %for.cond.cleanup3 ]
  %exitcond49.not = icmp eq i32 %i.0, %smax48
  br i1 %exitcond49.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %src_pool_ptr.0 = phi i32* [ %add.ptr, %for.cond.cleanup7 ], [ %src_ptr.0, %for.cond1.preheader ]
  %w.0 = phi i32 [ %w.1.lcssa, %for.cond.cleanup7 ], [ -32768, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc11, %for.cond.cleanup7 ], [ 0, %for.cond1.preheader ]
  %exitcond47.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond47.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond1
  br label %for.cond5

for.cond.cleanup3:                                ; preds = %for.cond1
  %w.0.lcssa = phi i32 [ %w.0, %for.cond1 ]
  %arrayidx13 = getelementptr inbounds i32, i32* %dest, i32 %i.0
  %2 = call i32 (i32, i32*, ...) @cgra_store32(i32 %w.0.lcssa, i32* %arrayidx13)
  %add.ptr14 = getelementptr inbounds i32, i32* %src_ptr.0, i32 %pool_size
  %inc15 = add nsw i32 %col.0, 1
  %cmp16 = icmp eq i32 %inc15, %output_cols
  %add.ptr18 = getelementptr inbounds i32, i32* %add.ptr14, i32 %input_rows_bump
  %spec.select = select i1 %cmp16, i32 0, i32 %inc15
  %spec.select45 = select i1 %cmp16, i32* %add.ptr18, i32* %add.ptr14
  %inc21 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.cond5:                                        ; preds = %for.body8, %for.cond5.preheader
  %w.1 = phi i32 [ %spec.select46, %for.body8 ], [ %w.0, %for.cond5.preheader ]
  %k.0 = phi i32 [ %inc, %for.body8 ], [ 0, %for.cond5.preheader ]
  %exitcond.not = icmp eq i32 %k.0, %pool_size
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.cond5
  %w.1.lcssa = phi i32 [ %w.1, %for.cond5 ]
  %add.ptr = getelementptr inbounds i32, i32* %src_pool_ptr.0, i32 %input_cols
  %inc11 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !6

for.body8:                                        ; preds = %for.cond5
  %arrayidx = getelementptr inbounds i32, i32* %src_pool_ptr.0, i32 %k.0
  %3 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx, i32 %lso.alloc.0)
  %cmp9 = icmp sgt i32 %3, %w.1
  %spec.select46 = select i1 %cmp9, i32 %3, i32 %w.1
  %inc = add nuw i32 %k.0, 1
  br label %for.cond5, !llvm.loop !7
}
