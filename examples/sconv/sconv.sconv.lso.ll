; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @sconv(i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %Brow, i32* noalias nocapture readonly %Bcol, i32* noalias nocapture readonly %B, i32* noalias nocapture %Z, i32 %rowBound, i32 %colBound, i32 %n, i32 %total_elements) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %total_elements, 0
  %smax = select i1 %0, i32 %total_elements, i32 0
  %1 = icmp sgt i32 %colBound, 0
  %smax47 = select i1 %1, i32 %colBound, i32 0
  %2 = icmp sgt i32 %rowBound, 0
  %smax49 = select i1 %2, i32 %rowBound, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc24, %for.cond.cleanup3 ]
  %row.0 = phi i32 [ 0, %entry ], [ %add22, %for.cond.cleanup3 ]
  %exitcond50.not = icmp eq i32 %i.0, %smax49
  br i1 %exitcond50.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %mul = mul nsw i32 %i.0, %n
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond.cleanup7, %for.body
  %offset.0 = phi i32 [ %mul, %for.body ], [ %inc18, %for.cond.cleanup7 ]
  %j.0 = phi i32 [ 0, %for.body ], [ %inc20, %for.cond.cleanup7 ]
  %exitcond48.not = icmp eq i32 %j.0, %smax47
  br i1 %exitcond48.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond1
  br label %for.cond5

for.cond.cleanup3:                                ; preds = %for.cond1
  %add22 = add nsw i32 %row.0, %colBound
  %inc24 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.cond5:                                        ; preds = %for.body8, %for.cond5.preheader
  %w.0 = phi i32 [ %add15, %for.body8 ], [ 0, %for.cond5.preheader ]
  %k.0 = phi i32 [ %inc, %for.body8 ], [ 0, %for.cond5.preheader ]
  %exitcond.not = icmp eq i32 %k.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.cond5
  %w.0.lcssa = phi i32 [ %w.0, %for.cond5 ]
  %add16 = add nsw i32 %j.0, %row.0
  %arrayidx17 = getelementptr inbounds i32, i32* %Z, i32 %add16
  store i32 %w.0.lcssa, i32* %arrayidx17, align 4, !tbaa !6
  %inc18 = add nsw i32 %offset.0, 1
  %inc20 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10

for.body8:                                        ; preds = %for.cond5
  %arrayidx = getelementptr inbounds i32, i32* %Brow, i32 %k.0
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !6
  %arrayidx9 = getelementptr inbounds i32, i32* %Bcol, i32 %k.0
  %4 = load i32, i32* %arrayidx9, align 4, !tbaa !6
  %arrayidx10 = getelementptr inbounds i32, i32* %B, i32 %k.0
  %5 = load i32, i32* %arrayidx10, align 4, !tbaa !6
  %mul11 = mul nsw i32 %3, %n
  %add = add nsw i32 %mul11, %offset.0
  %add12 = add nsw i32 %add, %4
  %arrayidx13 = getelementptr inbounds i32, i32* %A, i32 %add12
  %6 = load i32, i32* %arrayidx13, align 4, !tbaa !6
  %mul14 = mul nsw i32 %6, %5
  %add15 = add nsw i32 %mul14, %w.0
  %inc = add nuw i32 %k.0, 1
  br label %for.cond5, !llvm.loop !11
}
