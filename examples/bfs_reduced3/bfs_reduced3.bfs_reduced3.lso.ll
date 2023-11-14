; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @bfs_reduced3(i32* noalias nocapture %A, i32* noalias nocapture readonly %indices, i32 %len) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %len, 0
  %smax = select i1 %0, i32 %len, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup4, %entry
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %lso.alloc1.1.lcssa, %for.cond.cleanup4 ]
  %j.0 = phi i32 [ 0, %entry ], [ %inc7, %for.cond.cleanup4 ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %1 = call i32 (i32*, ...) @cgra_load32(i32* %A, i32 %lso.alloc1.0)
  %arrayidx = getelementptr inbounds i32, i32* %indices, i32 %1
  %2 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add = add nsw i32 %1, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %indices, i32 %add
  %3 = load i32, i32* %arrayidx2, align 4, !tbaa !4
  br label %for.cond1

for.cond1:                                        ; preds = %for.body5, %for.body
  %lso.alloc1.1 = phi i32 [ %lso.alloc1.0, %for.body ], [ %4, %for.body5 ]
  %i.0 = phi i32 [ %2, %for.body ], [ %inc, %for.body5 ]
  %cmp3 = icmp slt i32 %i.0, %3
  br i1 %cmp3, label %for.body5, label %for.cond.cleanup4

for.cond.cleanup4:                                ; preds = %for.cond1
  %lso.alloc1.1.lcssa = phi i32 [ %lso.alloc1.1, %for.cond1 ]
  %inc7 = add nuw i32 %j.0, 1
  br label %for.cond, !llvm.loop !8

for.body5:                                        ; preds = %for.cond1
  %4 = call i32 (i32, i32*, ...) @cgra_store32(i32 %i.0, i32* %A)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond1, !llvm.loop !10
}
