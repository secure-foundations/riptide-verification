; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @smm(i32* noalias nocapture readonly %Arow, i32* noalias nocapture readonly %Acol, i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %Brow, i32* noalias nocapture readonly %Bcol, i32* noalias nocapture readonly %B, i32* noalias nocapture %Z, i32 %rows, i32 %cols) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %rows, 0
  %smax = select i1 %0, i32 %rows, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup4, %entry
  %lso.alloc.0 = phi i32 [ 0, %entry ], [ %lso.alloc.1.lcssa, %for.cond.cleanup4 ]
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.cond.cleanup4 ]
  %dest_ptr.0 = phi i32* [ %Z, %entry ], [ %add.ptr21, %for.cond.cleanup4 ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %Arow, i32 %i.0
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add = add nuw i32 %i.0, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %Arow, i32 %add
  %2 = load i32, i32* %arrayidx1, align 4, !tbaa !4
  br label %for.cond2

for.cond2:                                        ; preds = %for.cond.cleanup13, %for.body
  %lso.alloc.1 = phi i32 [ %lso.alloc.0, %for.body ], [ %lso.alloc.2.lcssa, %for.cond.cleanup13 ]
  %j.0 = phi i32 [ %1, %for.body ], [ %inc19, %for.cond.cleanup13 ]
  %cmp3 = icmp slt i32 %j.0, %2
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %for.cond2
  %lso.alloc.1.lcssa = phi i32 [ %lso.alloc.1, %for.cond2 ]
  %add.ptr21 = getelementptr inbounds i32, i32* %dest_ptr.0, i32 %cols
  br label %for.cond, !llvm.loop !8

for.body5:                                        ; preds = %for.cond2
  %arrayidx6 = getelementptr inbounds i32, i32* %Acol, i32 %j.0
  %3 = load i32, i32* %arrayidx6, align 4, !tbaa !4
  %arrayidx7 = getelementptr inbounds i32, i32* %Brow, i32 %3
  %4 = load i32, i32* %arrayidx7, align 4, !tbaa !4
  %add8 = add nsw i32 %3, 1
  %arrayidx9 = getelementptr inbounds i32, i32* %Brow, i32 %add8
  %5 = load i32, i32* %arrayidx9, align 4, !tbaa !4
  %arrayidx10 = getelementptr inbounds i32, i32* %A, i32 %j.0
  %6 = load i32, i32* %arrayidx10, align 4, !tbaa !4
  br label %for.cond11

for.cond11:                                       ; preds = %for.body14, %for.body5
  %lso.alloc.2 = phi i32 [ %lso.alloc.1, %for.body5 ], [ %10, %for.body14 ]
  %k.0 = phi i32 [ %4, %for.body5 ], [ %inc, %for.body14 ]
  %cmp12 = icmp slt i32 %k.0, %5
  br i1 %cmp12, label %for.body14, label %for.cond.cleanup13

for.cond.cleanup13:                               ; preds = %for.cond11
  %lso.alloc.2.lcssa = phi i32 [ %lso.alloc.2, %for.cond11 ]
  %inc19 = add nsw i32 %j.0, 1
  br label %for.cond2, !llvm.loop !10

for.body14:                                       ; preds = %for.cond11
  %arrayidx15 = getelementptr inbounds i32, i32* %B, i32 %k.0
  %7 = load i32, i32* %arrayidx15, align 4, !tbaa !4
  %mul = mul nsw i32 %7, %6
  %arrayidx16 = getelementptr inbounds i32, i32* %Bcol, i32 %k.0
  %8 = load i32, i32* %arrayidx16, align 4, !tbaa !4
  %add.ptr = getelementptr inbounds i32, i32* %dest_ptr.0, i32 %8
  %9 = call i32 (i32*, ...) @cgra_load32(i32* %add.ptr, i32 %lso.alloc.2)
  %add17 = add nsw i32 %9, %mul
  %10 = call i32 (i32, i32*, ...) @cgra_store32(i32 %add17, i32* %add.ptr)
  %inc = add nsw i32 %k.0, 1
  br label %for.cond11, !llvm.loop !11
}
