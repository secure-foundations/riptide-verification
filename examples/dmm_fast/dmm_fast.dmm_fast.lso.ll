; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @dmm_fast(i32* noalias nocapture readonly %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %B_offset2, i32* noalias nocapture %Z, i32 %m, i32 %n, i32 %p, i32 %p_half) local_unnamed_addr #0 {
entry:
  %0 = icmp sgt i32 %n, 0
  %smax = select i1 %0, i32 %n, i32 0
  %1 = icmp sgt i32 %p_half, 0
  %smax53 = select i1 %1, i32 %p_half, i32 0
  %2 = icmp sgt i32 %m, 0
  %smax55 = select i1 %2, i32 %m, i32 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %lso.alloc1.1.lcssa, %for.cond.cleanup3 ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc23, %for.cond.cleanup3 ]
  %dest_ptr.0 = phi i32* [ %Z, %entry ], [ %add.ptr21, %for.cond.cleanup3 ]
  %filter_ptr.0 = phi i32* [ %A, %entry ], [ %add.ptr, %for.cond.cleanup3 ]
  %exitcond56.not = icmp eq i32 %i.0, %smax55
  br i1 %exitcond56.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond.cleanup:                                 ; preds = %for.cond
  ret void

for.cond1:                                        ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %lso.alloc1.1 = phi i32 [ %lso.alloc1.0, %for.cond1.preheader ], [ %5, %for.cond.cleanup7 ]
  %j.0 = phi i32 [ %inc19, %for.cond.cleanup7 ], [ 0, %for.cond1.preheader ]
  %exitcond54.not = icmp eq i32 %j.0, %smax53
  br i1 %exitcond54.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.cond1
  %lso.alloc1.1.lcssa = phi i32 [ %lso.alloc1.1, %for.cond1 ]
  %add.ptr = getelementptr inbounds i32, i32* %filter_ptr.0, i32 %n
  %add.ptr21 = getelementptr inbounds i32, i32* %dest_ptr.0, i32 %p
  %inc23 = add nuw i32 %i.0, 1
  br label %for.cond, !llvm.loop !4

for.body4:                                        ; preds = %for.cond1
  %3 = shl nuw i32 %j.0, 1
  %shl = and i32 %3, -4
  %and = and i32 %j.0, 1
  %add = or i32 %shl, %and
  br label %for.cond5

for.cond5:                                        ; preds = %for.body8, %for.body4
  %w.0 = phi i32 [ 0, %for.body4 ], [ %add10, %for.body8 ]
  %x.0 = phi i32 [ 0, %for.body4 ], [ %add13, %for.body8 ]
  %src_idx.0 = phi i32 [ %add, %for.body4 ], [ %add14, %for.body8 ]
  %k.0 = phi i32 [ 0, %for.body4 ], [ %inc, %for.body8 ]
  %exitcond.not = icmp eq i32 %k.0, %smax
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8

for.cond.cleanup7:                                ; preds = %for.cond5
  %w.0.lcssa = phi i32 [ %w.0, %for.cond5 ]
  %x.0.lcssa = phi i32 [ %x.0, %for.cond5 ]
  %arrayidx15 = getelementptr inbounds i32, i32* %dest_ptr.0, i32 %add
  %4 = call i32 (i32, i32*, ...) @cgra_store32(i32 %w.0.lcssa, i32* %arrayidx15, i32 %lso.alloc1.1)
  %add16 = or i32 %add, 2
  %arrayidx17 = getelementptr inbounds i32, i32* %dest_ptr.0, i32 %add16
  %5 = call i32 (i32, i32*, ...) @cgra_store32(i32 %x.0.lcssa, i32* %arrayidx17, i32 %4)
  %inc19 = add nuw i32 %j.0, 1
  br label %for.cond1, !llvm.loop !6

for.body8:                                        ; preds = %for.cond5
  %arrayidx = getelementptr inbounds i32, i32* %filter_ptr.0, i32 %k.0
  %6 = load i32, i32* %arrayidx, align 4, !tbaa !7
  %arrayidx9 = getelementptr inbounds i32, i32* %B, i32 %src_idx.0
  %7 = load i32, i32* %arrayidx9, align 4, !tbaa !7
  %mul = mul nsw i32 %7, %6
  %add10 = add nsw i32 %mul, %w.0
  %arrayidx11 = getelementptr inbounds i32, i32* %B_offset2, i32 %src_idx.0
  %8 = load i32, i32* %arrayidx11, align 4, !tbaa !7
  %mul12 = mul nsw i32 %8, %6
  %add13 = add nsw i32 %mul12, %x.0
  %add14 = add nsw i32 %src_idx.0, %p
  %inc = add nuw i32 %k.0, 1
  br label %for.cond5, !llvm.loop !11
}
