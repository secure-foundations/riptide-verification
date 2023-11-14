; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @nn_normalize(i32* nocapture readonly %src, i32* nocapture %dest, i32 %size, i32 %max, i32 %shift) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %size, 0
  %smax = select i1 %0, i32 %size, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %2, %for.body ]
  %src_ptr.0 = phi i32* [ %src, %entry ], [ %incdec.ptr, %for.body ]
  %dest_ptr.0 = phi i32* [ %dest, %entry ], [ %incdec.ptr1, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %incdec.ptr = getelementptr inbounds i32, i32* %src_ptr.0, i32 1
  %1 = call i32 (i32*, ...) @cgra_load32(i32* %src_ptr.0, i32 %lso.alloc1.0)
  %mul = mul nsw i32 %1, %max
  %shr = ashr i32 %mul, %shift
  %incdec.ptr1 = getelementptr inbounds i32, i32* %dest_ptr.0, i32 1
  %2 = call i32 (i32, i32*, ...) @cgra_store32(i32 %shr, i32* %dest_ptr.0)
  %inc = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4
}
