; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @dconv(i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %B, i32* noalias nocapture %Z, i32 %size, i32 %cols, i32 %scols, i32 %frows, i32 %fcols) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %fcols, 0
  %smax = select i1 %0, i32 %fcols, i32 0
  %1 = icmp sgt i32 %frows, 0
  %smax49 = select i1 %1, i32 %frows, i32 0
  %2 = icmp sgt i32 %size, 0
  %smax51 = select i1 %2, i32 %size, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %col.0 = phi i32 [ 0, %entry ], [ %spec.select, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc25, %for.cond.cleanup3 ]
  %row.0 = phi i32 [ 0, %entry ], [ %spec.select48, %for.cond.cleanup3 ]
  %exitcond52.not = icmp eq i32 %i.0, %smax51
  br i1 %exitcond52.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %w.0 = phi i32 [ %w.1.lcssa, %for.cond.cleanup7 ], [ 0, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc18, %for.cond.cleanup7 ], [ 0, %for.cond1.preheader ]
  %exitcond50.not = icmp eq i32 %j.0, %smax49
  br i1 %exitcond50.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond1
  %reass.add = add nuw i32 %j.0, %row.0
  %reass.mul = mul i32 %reass.add, %scols
  %add10 = add i32 %reass.mul, %col.0
  %mul12 = mul nsw i32 %j.0, %frows
  br label %for.cond5

for.cond.cleanup3:                                ; preds = %for.cond1
  %w.0.lcssa = phi i32 [ %w.0, %for.cond1 ]
  %arrayidx20 = getelementptr inbounds i32, i32* %Z, i32 %i.0
  store i32 %w.0.lcssa, i32* %arrayidx20, align 4, !tbaa !4
  %inc21 = add nsw i32 %col.0, 1
  %cmp22 = icmp eq i32 %inc21, %cols
  %spec.select = select i1 %cmp22, i32 0, i32 %inc21
  %inc23 = zext i1 %cmp22 to i32
  %spec.select48 = add nuw nsw i32 %row.0, %inc23
  %inc25 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !8

for.cond5:                                        ; preds = %for.body8, %for.cond5.preheader
  %w.1 = phi i32 [ %add16, %for.body8 ], [ %w.0, %for.cond5.preheader ]
  %k.0 = phi i32 [ %inc, %for.body8 ], [ 0, %for.cond5.preheader ]
  %exitcond.not = icmp eq i32 %k.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.cond5
  %w.1.lcssa = phi i32 [ %w.1, %for.cond5 ]
  %inc18 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10

for.body8:                                        ; preds = %for.cond5
  %add11 = add i32 %add10, %k.0
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %add11
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !4
  %add13 = add nsw i32 %k.0, %mul12
  %arrayidx14 = getelementptr inbounds i32, i32* %B, i32 %add13
  %4 = load i32, i32* %arrayidx14, align 4, !tbaa !4
  %mul15 = mul nsw i32 %4, %3
  %add16 = add nsw i32 %mul15, %w.1
  %inc = add nuw i32 %k.0, 1
  br label %for.cond5, !llvm.loop !11
}
