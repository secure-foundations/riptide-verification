; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_pool(i32* noalias nocapture readonly %src, i32* noalias nocapture %dest, i32 %input_rows_bump, i32 %input_cols, i32 %output_size, i32 %output_cols, i32 %pool_size) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %pool_size, 0
  %smax = select i1 %0, i32 %pool_size, i32 0
  %1 = icmp sgt i32 %output_size, 0
  %smax47 = select i1 %1, i32 %output_size, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %col.0 = phi i32 [ 0, %entry ], [ %spec.select, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc22, %for.cond.cleanup3 ]
  %src_offset.0 = phi i32 [ 0, %entry ], [ %spec.select44, %for.cond.cleanup3 ]
  %exitcond48.not = icmp eq i32 %i.0, %smax47
  br i1 %exitcond48.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %w.0 = phi i32 [ %w.1.lcssa, %for.cond.cleanup7 ], [ -32768, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc12, %for.cond.cleanup7 ], [ 0, %for.cond1.preheader ]
  %exitcond46.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond46.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond1
  %mul = mul nsw i32 %j.0, %input_cols
  %add = add nsw i32 %mul, %src_offset.0
  br label %for.cond5

for.cond.cleanup3:                                ; preds = %for.cond1
  %w.0.lcssa = phi i32 [ %w.0, %for.cond1 ]
  %arrayidx14 = getelementptr inbounds i32, i32* %dest, i32 %i.0
  store i32 %w.0.lcssa, i32* %arrayidx14, align 4, !tbaa !4
  %add15 = add nsw i32 %src_offset.0, %pool_size
  %inc16 = add nsw i32 %col.0, 1
  %cmp17 = icmp eq i32 %inc16, %output_cols
  %spec.select = select i1 %cmp17, i32 0, i32 %inc16
  %add19 = select i1 %cmp17, i32 %input_rows_bump, i32 0
  %spec.select44 = add nsw i32 %add15, %add19
  %inc22 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8

for.cond5:                                        ; preds = %for.body8, %for.cond5.preheader
  %w.1 = phi i32 [ %spec.select45, %for.body8 ], [ %w.0, %for.cond5.preheader ]
  %k.0 = phi i32 [ %inc, %for.body8 ], [ 0, %for.cond5.preheader ]
  %exitcond.not = icmp eq i32 %k.0, %pool_size
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.cond5
  %w.1.lcssa = phi i32 [ %w.1, %for.cond5 ]
  %inc12 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10

for.body8:                                        ; preds = %for.cond5
  %add9 = add nsw i32 %add, %k.0
  %arrayidx = getelementptr inbounds i32, i32* %src, i32 %add9
  %2 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %cmp10 = icmp sgt i32 %2, %w.1
  %spec.select45 = select i1 %cmp10, i32 %2, i32 %w.1
  %inc = add nuw i32 %k.0, 1
  br label %for.cond5, !llvm.loop !11
}
