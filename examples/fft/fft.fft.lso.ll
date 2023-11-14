; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @fft(i32* noalias nocapture %real, i32* noalias nocapture %imag, i32* noalias nocapture readonly %real_twiddle, i32* noalias nocapture readonly %imag_twiddle, i32 %size, i32 %stride, i32 %step, i32 %Ls, i32 %theta, i32 %strided_step, i32 %Ls_stride) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %Ls, 0
  %smax = select i1 %0, i32 %Ls, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup4, %entry
  %lso.alloc4.0 = phi i32 [ 0, %entry ], [ %lso.alloc4.1.lcssa, %for.cond.cleanup4 ]
  %lso.alloc5.0 = phi i32 [ 0, %entry ], [ %lso.alloc5.1.lcssa, %for.cond.cleanup4 ]
  %lso.alloc3.0 = phi i32 [ 0, %entry ], [ %lso.alloc3.1.lcssa, %for.cond.cleanup4 ]
  %lso.alloc7.0 = phi i32 [ 0, %entry ], [ %lso.alloc7.1.lcssa, %for.cond.cleanup4 ]
  %lso.alloc8.0 = phi i32 [ 0, %entry ], [ %lso.alloc8.1.lcssa, %for.cond.cleanup4 ]
  %lso.alloc9.0 = phi i32 [ 0, %entry ], [ %lso.alloc9.1.lcssa, %for.cond.cleanup4 ]
  %j.0 = phi i32 [ 0, %entry ], [ %inc, %for.cond.cleanup4 ]
  %imag.addr.0 = phi i32* [ %imag, %entry ], [ %imag.addr.1.lcssa, %for.cond.cleanup4 ]
  %real.addr.0 = phi i32* [ %real, %entry ], [ %real.addr.1.lcssa, %for.cond.cleanup4 ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %j.0, %theta
  %arrayidx = getelementptr inbounds i32, i32* %real_twiddle, i32 %add
  %1 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx, i32 undef, i32 %lso.alloc9.0)
  %arrayidx1 = getelementptr inbounds i32, i32* %imag_twiddle, i32 %add
  %2 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx1, i32 undef, i32 %lso.alloc8.0)
  %mul = mul nsw i32 %j.0, %stride
  %add6 = add nsw i32 %mul, %Ls_stride
  br label %for.cond2

for.cond2:                                        ; preds = %for.body5, %for.body
  %lso.alloc4.1 = phi i32 [ %lso.alloc4.0, %for.body ], [ %9, %for.body5 ]
  %lso.alloc5.1 = phi i32 [ %lso.alloc5.0, %for.body ], [ %10, %for.body5 ]
  %lso.alloc3.1 = phi i32 [ %lso.alloc3.0, %for.body ], [ %10, %for.body5 ]
  %lso.alloc7.1 = phi i32 [ %lso.alloc7.0, %for.body ], [ %9, %for.body5 ]
  %lso.alloc8.1 = phi i32 [ %lso.alloc8.0, %for.body ], [ %10, %for.body5 ]
  %lso.alloc9.1 = phi i32 [ %lso.alloc9.0, %for.body ], [ %10, %for.body5 ]
  %k.0 = phi i32 [ %j.0, %for.body ], [ %add35, %for.body5 ]
  %imag.addr.1 = phi i32* [ %imag.addr.0, %for.body ], [ %add.ptr34, %for.body5 ]
  %real.addr.1 = phi i32* [ %real.addr.0, %for.body ], [ %add.ptr, %for.body5 ]
  %cmp3 = icmp slt i32 %k.0, %size
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %for.cond2
  %lso.alloc4.1.lcssa = phi i32 [ %lso.alloc4.1, %for.cond2 ]
  %lso.alloc5.1.lcssa = phi i32 [ %lso.alloc5.1, %for.cond2 ]
  %lso.alloc3.1.lcssa = phi i32 [ %lso.alloc3.1, %for.cond2 ]
  %lso.alloc7.1.lcssa = phi i32 [ %lso.alloc7.1, %for.cond2 ]
  %lso.alloc8.1.lcssa = phi i32 [ %lso.alloc8.1, %for.cond2 ]
  %lso.alloc9.1.lcssa = phi i32 [ %lso.alloc9.1, %for.cond2 ]
  %imag.addr.1.lcssa = phi i32* [ %imag.addr.1, %for.cond2 ]
  %real.addr.1.lcssa = phi i32* [ %real.addr.1, %for.cond2 ]
  %inc = add nuw i32 %j.0, 1
  br label %for.cond, !llvm.loop !4

for.body5:                                        ; preds = %for.cond2
  %arrayidx7 = getelementptr inbounds i32, i32* %real.addr.1, i32 %add6
  %3 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx7, i32 %lso.alloc7.1)
  %arrayidx10 = getelementptr inbounds i32, i32* %imag.addr.1, i32 %add6
  %4 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx10, i32 %lso.alloc5.1)
  %mul11 = mul nsw i32 %3, %1
  %mul12 = mul nsw i32 %4, %2
  %sub = sub nsw i32 %mul11, %mul12
  %mul13 = mul nsw i32 %4, %1
  %mul14 = mul nsw i32 %3, %2
  %add15 = add nsw i32 %mul13, %mul14
  %arrayidx17 = getelementptr inbounds i32, i32* %real.addr.1, i32 %mul
  %5 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx17, i32 %lso.alloc4.1)
  %arrayidx19 = getelementptr inbounds i32, i32* %imag.addr.1, i32 %mul
  %6 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx19, i32 %lso.alloc3.1)
  %sub20 = sub nsw i32 %5, %sub
  %7 = call i32 (i32, i32*, ...) @cgra_store32(i32 %sub20, i32* %arrayidx7)
  %sub24 = sub nsw i32 %6, %add15
  %8 = call i32 (i32, i32*, ...) @cgra_store32(i32 %sub24, i32* %arrayidx10)
  %add28 = add nsw i32 %sub, %5
  %9 = call i32 (i32, i32*, ...) @cgra_store32(i32 %add28, i32* %arrayidx17, i32 %7)
  %add31 = add nsw i32 %6, %add15
  %10 = call i32 (i32, i32*, ...) @cgra_store32(i32 %add31, i32* %arrayidx19, i32 %8)
  %add.ptr = getelementptr inbounds i32, i32* %real.addr.1, i32 %strided_step
  %add.ptr34 = getelementptr inbounds i32, i32* %imag.addr.1, i32 %strided_step
  %add35 = add nsw i32 %k.0, %step
  br label %for.cond2, !llvm.loop !6
}
