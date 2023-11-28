; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @SpMSpMd(i32* nocapture %Y, i32 %R, i32 %C, i32* noalias nocapture readonly %aa, i32* noalias nocapture readonly %aj, i32* noalias nocapture readonly %ai, i32* noalias nocapture readonly %ba, i32* noalias nocapture readonly %bj, i32* noalias nocapture readonly %bi) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %C, 0
  %smax = select i1 %0, i32 %C, i32 0
  %1 = icmp sgt i32 %R, 0
  %smax74 = select i1 %1, i32 %R, i32 0
  br label %for.cond

for.cond.loopexit:                                ; preds = %for.cond2
  br label %for.cond, !llvm.loop !4

for.cond:                                         ; preds = %for.cond.loopexit, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %add, %for.cond.loopexit ]
  %exitcond75.not = icmp eq i32 %i.0, %smax74
  br i1 %exitcond75.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %ai, i32 %i.0
  %2 = load i32, i32* %arrayidx, align 4, !tbaa !6
  %add = add nuw i32 %i.0, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %ai, i32 %add
  %3 = load i32, i32* %arrayidx1, align 4, !tbaa !6
  %mul29 = mul nsw i32 %i.0, %C
  br label %for.cond2

for.cond2:                                        ; preds = %while.end, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %add7, %while.end ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.cond.loopexit, label %for.body5

for.body5:                                        ; preds = %for.cond2
  %arrayidx6 = getelementptr inbounds i32, i32* %bi, i32 %j.0
  %4 = load i32, i32* %arrayidx6, align 4, !tbaa !6
  %add7 = add nuw i32 %j.0, 1
  %arrayidx8 = getelementptr inbounds i32, i32* %bi, i32 %add7
  %5 = load i32, i32* %arrayidx8, align 4, !tbaa !6
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body5
  %new_iA.0 = phi i32 [ %2, %for.body5 ], [ %spec.select, %if.end ]
  %new_iB.0 = phi i32 [ %4, %for.body5 ], [ %new_iB.1, %if.end ]
  %acc.0 = phi i32 [ 0, %for.body5 ], [ %acc.1, %if.end ]
  %cmp9 = icmp slt i32 %new_iA.0, %3
  %cmp10 = icmp slt i32 %new_iB.0, %5
  %6 = and i1 %cmp9, %cmp10
  br i1 %6, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %arrayidx11 = getelementptr inbounds i32, i32* %aj, i32 %new_iA.0
  %7 = load i32, i32* %arrayidx11, align 4, !tbaa !6
  %arrayidx12 = getelementptr inbounds i32, i32* %bj, i32 %new_iB.0
  %8 = load i32, i32* %arrayidx12, align 4, !tbaa !6
  %cmp13 = icmp eq i32 %7, %8
  br i1 %cmp13, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %arrayidx14 = getelementptr inbounds i32, i32* %aa, i32 %new_iA.0
  %9 = load i32, i32* %arrayidx14, align 4, !tbaa !6
  %arrayidx15 = getelementptr inbounds i32, i32* %ba, i32 %new_iB.0
  %10 = load i32, i32* %arrayidx15, align 4, !tbaa !6
  %mul = mul nsw i32 %10, %9
  %add16 = add nsw i32 %mul, %acc.0
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  %acc.1 = phi i32 [ %add16, %if.then ], [ %acc.0, %while.body ]
  %cmp19.not = icmp sle i32 %7, %8
  %add21 = zext i1 %cmp19.not to i32
  %spec.select = add nsw i32 %new_iA.0, %add21
  %cmp25.not = icmp sge i32 %7, %8
  %add27 = zext i1 %cmp25.not to i32
  %new_iB.1 = add nsw i32 %new_iB.0, %add27
  br label %while.cond, !llvm.loop !10

while.end:                                        ; preds = %while.cond
  %acc.0.lcssa = phi i32 [ %acc.0, %while.cond ]
  %add30 = add nsw i32 %j.0, %mul29
  %arrayidx31 = getelementptr inbounds i32, i32* %Y, i32 %add30
  store i32 %acc.0.lcssa, i32* %arrayidx31, align 4, !tbaa !6
  br label %for.cond2, !llvm.loop !11
}
