; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @dmm(i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %B, i32* nocapture %Z, i32 %m, i32 %n, i32 %p) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %n, 0
  %smax = select i1 %0, i32 %n, i32 0
  %1 = icmp sgt i32 %p, 0
  %smax36 = select i1 %1, i32 %p, i32 0
  %2 = icmp sgt i32 %m, 0
  %smax38 = select i1 %2, i32 %m, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %dest_idx.0 = phi i32 [ 0, %entry ], [ %dest_idx.1.lcssa, %for.cond.cleanup3 ]
  %filter_ptr.0 = phi i32* [ %A, %entry ], [ %add.ptr, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc17, %for.cond.cleanup3 ]
  %exitcond39.not = icmp eq i32 %i.0, %smax38
  br i1 %exitcond39.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %dest_idx.1 = phi i32 [ %inc11, %for.cond.cleanup7 ], [ %dest_idx.0, %for.cond1.preheader ]
  %j.0 = phi i32 [ %inc14, %for.cond.cleanup7 ], [ 0, %for.cond1.preheader ]
  %exitcond37.not = icmp eq i32 %j.0, %smax36
  br i1 %exitcond37.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.cond1
  br label %for.cond5

for.cond.cleanup3:                                ; preds = %for.cond1
  %dest_idx.1.lcssa = phi i32 [ %dest_idx.1, %for.cond1 ]
  %add.ptr = getelementptr inbounds i32, i32* %filter_ptr.0, i32 %n
  %inc17 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.cond5:                                        ; preds = %for.body8, %for.cond5.preheader
  %w.0 = phi i32 [ %add, %for.body8 ], [ 0, %for.cond5.preheader ]
  %src_idx.0 = phi i32 [ %add10, %for.body8 ], [ %j.0, %for.cond5.preheader ]
  %k.0 = phi i32 [ %inc, %for.body8 ], [ 0, %for.cond5.preheader ]
  %exitcond.not = icmp eq i32 %k.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.cond5
  %w.0.lcssa = phi i32 [ %w.0, %for.cond5 ]
  %inc11 = add nsw i32 %dest_idx.1, 1
  %arrayidx12 = getelementptr inbounds i32, i32* %Z, i32 %dest_idx.1
  store i32 %w.0.lcssa, i32* %arrayidx12, align 4, !tbaa !6
  %inc14 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !10

for.body8:                                        ; preds = %for.cond5
  %arrayidx = getelementptr inbounds i32, i32* %filter_ptr.0, i32 %k.0
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !6
  %arrayidx9 = getelementptr inbounds i32, i32* %B, i32 %src_idx.0
  %4 = load i32, i32* %arrayidx9, align 4, !tbaa !6
  %mul = mul nsw i32 %4, %3
  %add = add nsw i32 %mul, %w.0
  %add10 = add nsw i32 %src_idx.0, %p
  %inc = add nuw i32 %k.0, 1
  br label %for.cond5, !llvm.loop !11
}
