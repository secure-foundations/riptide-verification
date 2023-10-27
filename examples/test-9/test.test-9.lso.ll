; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @test(i32* nocapture %A, i32* nocapture %B, i32 %len) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %len, 0
  %smax34 = select i1 %0, i32 %len, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %lso.alloc.0 = phi i32 [ 0, %entry ], [ %lso.alloc.1.lcssa, %for.cond.cleanup3 ]
  %lso.alloc3.0 = phi i32 [ 0, %entry ], [ %lso.alloc3.1.lcssa, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc8, %for.cond.cleanup3 ]
  %exitcond35.not = icmp eq i32 %i.0, %smax34
  br i1 %exitcond35.not, label %for.cond11.preheader, label %for.body

for.cond11.preheader:                             ; preds = %for.cond
  %lso.alloc.0.lcssa = phi i32 [ %lso.alloc.0, %for.cond ]
  br label %for.cond11

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.0
  %1 = call i32 (i32, i32*, ...) @cgra_store32(i32 0, i32* %arrayidx, i32 undef, i32 %lso.alloc3.0)
  br label %for.cond1

for.cond1:                                        ; preds = %for.body4, %for.body
  %lso.alloc.1 = phi i32 [ %lso.alloc.0, %for.body ], [ %4, %for.body4 ]
  %lso.alloc3.1 = phi i32 [ %lso.alloc3.0, %for.body ], [ %4, %for.body4 ]
  %2 = phi i32 [ 0, %for.body ], [ %add, %for.body4 ]
  %j.0 = phi i32 [ %i.0, %for.body ], [ %inc, %for.body4 ]
  %cmp2 = icmp slt i32 %j.0, %len
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.cond1
  %lso.alloc.1.lcssa = phi i32 [ %lso.alloc.1, %for.cond1 ]
  %lso.alloc3.1.lcssa = phi i32 [ %lso.alloc3.1, %for.cond1 ]
  %inc8 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.body4:                                        ; preds = %for.cond1
  %arrayidx5 = getelementptr inbounds i32, i32* %B, i32 %j.0
  %3 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx5, i32 %1)
  %add = add nsw i32 %2, %3
  %4 = call i32 (i32, i32*, ...) @cgra_store32(i32 %add, i32* %arrayidx)
  %inc = add nuw nsw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !6

for.cond11:                                       ; preds = %for.body14, %for.cond11.preheader
  %i10.0 = phi i32 [ %inc17, %for.body14 ], [ 0, %for.cond11.preheader ]
  %exitcond.not = icmp eq i32 %i10.0, %smax34
  br i1 %exitcond.not, label %for.cond.cleanup13, label %for.body14

for.cond.cleanup13:                               ; preds = %for.cond11
  ret void

for.body14:                                       ; preds = %for.cond11
  %arrayidx15 = getelementptr inbounds i32, i32* %B, i32 %i10.0
  %5 = call i32 (i32, i32*, ...) @cgra_store32(i32 1, i32* %arrayidx15, i32 undef, i32 %lso.alloc.0.lcssa)
  %inc17 = add nuw i32 %i10.0, 1
  br label %for.cond11, !llvm.loop !7
}
