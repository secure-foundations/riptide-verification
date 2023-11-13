; Function Attrs: minsize nofree norecurse nounwind optsize ssp writeonly
define dso_local void @sort_reduced(i32* noalias nocapture %A, i32* noalias nocapture %Z, i32 %size) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %size, 0
  %smax = select i1 %0, i32 %size, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.inc14, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc15, %for.inc14 ]
  %exitcond31.not = icmp eq i32 %i.0, 32
  br i1 %exitcond31.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.body:                                         ; preds = %for.cond
  %and = and i32 %i.0, 1
  %tobool.not = icmp eq i32 %and, 0
  br i1 %tobool.not, label %for.cond6.preheader, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.body
  br label %for.cond1

for.cond6.preheader:                              ; preds = %for.body
  br label %for.cond6

for.cond1:                                        ; preds = %for.body4, %for.cond1.preheader
  %j.0 = phi i32 [ %inc, %for.body4 ], [ 0, %for.cond1.preheader ]
  %exitcond.not = icmp eq i32 %j.0, %smax
  br i1 %exitcond.not, label %for.inc14.loopexit2, label %for.body4

for.body4:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %j.0
  store i32 0, i32* %arrayidx, align 4, !tbaa !4
  %inc = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !8

for.cond6:                                        ; preds = %for.body9, %for.cond6.preheader
  %j5.0 = phi i32 [ %inc12, %for.body9 ], [ 0, %for.cond6.preheader ]
  %exitcond30.not = icmp eq i32 %j5.0, %smax
  br i1 %exitcond30.not, label %for.inc14.loopexit, label %for.body9

for.body9:                                        ; preds = %for.cond6
  %arrayidx10 = getelementptr inbounds i32, i32* %Z, i32 %j5.0
  store i32 0, i32* %arrayidx10, align 4, !tbaa !4
  %inc12 = add nuw i32 %j5.0, 1
  br label %for.cond6, !llvm.loop !10

for.inc14.loopexit:                               ; preds = %for.cond6
  br label %for.inc14

for.inc14.loopexit2:                              ; preds = %for.cond1
  br label %for.inc14

for.inc14:                                        ; preds = %for.inc14.loopexit2, %for.inc14.loopexit
  %inc15 = add nuw nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !11
}
