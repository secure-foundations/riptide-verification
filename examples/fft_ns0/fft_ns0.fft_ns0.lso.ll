; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @fft_ns0(i32* noalias nocapture readonly %src_real_ptr, i32* noalias nocapture readonly %src_imag_ptr, i32* noalias nocapture %dst_real_ptr, i32* noalias nocapture %dst_imag_ptr, i32 %size, i32 %stride, i32 %i_2, i32 %i_1, i32 %mask) local_unnamed_addr #0 {
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
  %add = add nsw i32 %i.0, %i_2
  %and = and i32 %add, %i_1
  %and1 = and i32 %i.0, %mask
  %add2 = add nsw i32 %and, %and1
  %mul = mul nsw i32 %i.0, %stride
  %arrayidx = getelementptr inbounds i32, i32* %src_real_ptr, i32 %mul
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %arrayidx3 = getelementptr inbounds i32, i32* %dst_real_ptr, i32 %add2
  store i32 %1, i32* %arrayidx3, align 4, !tbaa !4
  %arrayidx4 = getelementptr inbounds i32, i32* %src_imag_ptr, i32 %mul
  %2 = load i32, i32* %arrayidx4, align 4, !tbaa !4
  %arrayidx5 = getelementptr inbounds i32, i32* %dst_imag_ptr, i32 %add2
  store i32 %2, i32* %arrayidx5, align 4, !tbaa !4
  %inc = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8
}
