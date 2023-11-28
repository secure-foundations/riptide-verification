; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @SpSlice(i32* noalias nocapture readonly %ia, i32* noalias nocapture readonly %ij, i32* noalias nocapture readonly %ii, i32* noalias nocapture readonly %idxj, i32* noalias nocapture readonly %idxi, i32 %rows, i32* noalias nocapture %out) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %rows, 0
  %smax = select i1 %0, i32 %rows, i32 0
  br label %for.cond

for.cond.loopexit:                                ; preds = %while.cond
  br label %for.cond, !llvm.loop !4

for.cond:                                         ; preds = %for.cond.loopexit, %entry
  %row.0 = phi i32 [ 0, %entry ], [ %add, %for.cond.loopexit ]
  %exitcond.not = icmp eq i32 %row.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %ii, i32 %row.0
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !6
  %add = add nuw i32 %row.0, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %ii, i32 %add
  %2 = load i32, i32* %arrayidx1, align 4, !tbaa !6
  %arrayidx2 = getelementptr inbounds i32, i32* %idxi, i32 %row.0
  %3 = load i32, i32* %arrayidx2, align 4, !tbaa !6
  %arrayidx4 = getelementptr inbounds i32, i32* %idxi, i32 %add
  %4 = load i32, i32* %arrayidx4, align 4, !tbaa !6
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body
  %new_im.0 = phi i32 [ %1, %for.body ], [ %new_im.2, %if.end ]
  %new_ix.0 = phi i32 [ %3, %for.body ], [ %new_ix.2, %if.end ]
  %cmp5 = icmp slt i32 %new_im.0, %2
  %cmp6 = icmp slt i32 %new_ix.0, %4
  %5 = and i1 %cmp5, %cmp6
  br i1 %5, label %while.body, label %for.cond.loopexit

while.body:                                       ; preds = %while.cond
  %arrayidx7 = getelementptr inbounds i32, i32* %ij, i32 %new_im.0
  %6 = load i32, i32* %arrayidx7, align 4, !tbaa !6
  %arrayidx8 = getelementptr inbounds i32, i32* %idxj, i32 %new_ix.0
  %7 = load i32, i32* %arrayidx8, align 4, !tbaa !6
  %cmp9 = icmp eq i32 %6, %7
  br i1 %cmp9, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %arrayidx10 = getelementptr inbounds i32, i32* %ia, i32 %new_im.0
  %8 = load i32, i32* %arrayidx10, align 4, !tbaa !6
  %arrayidx11 = getelementptr inbounds i32, i32* %out, i32 %new_ix.0
  store i32 %8, i32* %arrayidx11, align 4, !tbaa !6
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  %cmp20.not = icmp slt i32 %6, %7
  %spec.select = zext i1 %cmp20.not to i32
  %new_im.2 = add nsw i32 %new_im.0, %spec.select
  %not.cmp20.not = xor i1 %cmp20.not, true
  %add22 = zext i1 %not.cmp20.not to i32
  %new_ix.2 = add nsw i32 %new_ix.0, %add22
  br label %while.cond, !llvm.loop !10
}
