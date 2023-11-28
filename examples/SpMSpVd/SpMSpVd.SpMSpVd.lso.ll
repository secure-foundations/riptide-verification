; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @SpMSpVd(i32* noalias nocapture %Y, i32 %R, i32 %C, i32* noalias nocapture readonly %aa, i32* noalias nocapture readonly %aj, i32* noalias nocapture readonly %ai, i32* noalias nocapture readonly %ba, i32* noalias nocapture readonly %bj, i32 %bnnz) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %R, 0
  %smax = select i1 %0, i32 %R, i32 0
  br label %for.cond

for.cond:                                         ; preds = %while.end, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add, %while.end ]
  %exitcond.not = icmp eq i32 %i.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %ai, i32 %i.0
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add = add nuw i32 %i.0, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %ai, i32 %add
  %2 = load i32, i32* %arrayidx1, align 4, !tbaa !4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body
  %acc.0 = phi i32 [ 0, %for.body ], [ %acc.1, %if.end ]
  %iB.0 = phi i32 [ 0, %for.body ], [ %new_iB.1, %if.end ]
  %iA.0 = phi i32 [ %1, %for.body ], [ %spec.select, %if.end ]
  %cmp2 = icmp slt i32 %iA.0, %2
  %cmp3 = icmp slt i32 %iB.0, %bnnz
  %3 = and i1 %cmp3, %cmp2
  br i1 %3, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %arrayidx4 = getelementptr inbounds i32, i32* %aj, i32 %iA.0
  %4 = load i32, i32* %arrayidx4, align 4, !tbaa !4
  %arrayidx5 = getelementptr inbounds i32, i32* %bj, i32 %iB.0
  %5 = load i32, i32* %arrayidx5, align 4, !tbaa !4
  %cmp8 = icmp eq i32 %4, %5
  br i1 %cmp8, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %arrayidx7 = getelementptr inbounds i32, i32* %ba, i32 %iB.0
  %6 = load i32, i32* %arrayidx7, align 4, !tbaa !4
  %arrayidx6 = getelementptr inbounds i32, i32* %aa, i32 %iA.0
  %7 = load i32, i32* %arrayidx6, align 4, !tbaa !4
  %mul = mul nsw i32 %7, %6
  %add9 = add nsw i32 %mul, %acc.0
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  %acc.1 = phi i32 [ %add9, %if.then ], [ %acc.0, %while.body ]
  %cmp10.not = icmp sle i32 %4, %5
  %add12 = zext i1 %cmp10.not to i32
  %spec.select = add nsw i32 %iA.0, %add12
  %cmp14.not = icmp sge i32 %4, %5
  %add16 = zext i1 %cmp14.not to i32
  %new_iB.1 = add nuw nsw i32 %iB.0, %add16
  br label %while.cond, !llvm.loop !8

while.end:                                        ; preds = %while.cond
  %acc.0.lcssa = phi i32 [ %acc.0, %while.cond ]
  %arrayidx18 = getelementptr inbounds i32, i32* %Y, i32 %i.0
  store i32 %acc.0.lcssa, i32* %arrayidx18, align 4, !tbaa !4
  br label %for.cond, !llvm.loop !10
}
