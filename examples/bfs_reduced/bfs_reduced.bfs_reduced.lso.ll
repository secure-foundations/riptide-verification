; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @bfs_reduced(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32 %len) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %len, 0
  %smax = select i1 %0, i32 %len, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %lso.alloc1.1, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.0
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 %1
  %2 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx1, i32 %lso.alloc1.0)
  %tobool.not = icmp eq i32 %2, 0
  br i1 %tobool.not, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %3 = call i32 (i32, i32*, ...) @cgra_store32(i32 1, i32* %arrayidx1)
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %lso.alloc1.1 = phi i32 [ %3, %if.then ], [ %lso.alloc1.0, %for.body ]
  %inc = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8
}
