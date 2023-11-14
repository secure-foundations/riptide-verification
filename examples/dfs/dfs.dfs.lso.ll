; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @dfs(i32* noalias nocapture readonly %rows, i32* noalias nocapture readonly %cols, i32 %count, i32* nocapture %stack, i32* nocapture %visited, i32* noalias nocapture %walk) local_unnamed_addr #0 {
entry:
  br label %while.cond

while.cond.loopexit:                              ; preds = %for.cond
  %lso.alloc2.1.lcssa = phi i32 [ %lso.alloc2.1, %for.cond ]
  %lso.alloc1.1.lcssa = phi i32 [ %lso.alloc1.1, %for.cond ]
  %stack_pos.1.lcssa = phi i32 [ %stack_pos.1, %for.cond ]
  %inc = add nuw nsw i32 %walk_pos.0, 1
  br label %while.cond, !llvm.loop !4

while.cond:                                       ; preds = %while.cond.loopexit, %entry
  %lso.alloc2.0 = phi i32 [ 0, %entry ], [ %lso.alloc2.1.lcssa, %while.cond.loopexit ]
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %lso.alloc1.1.lcssa, %while.cond.loopexit ]
  %walk_pos.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond.loopexit ]
  %stack_pos.0 = phi i32 [ 1, %entry ], [ %stack_pos.1.lcssa, %while.cond.loopexit ]
  %cmp.not = icmp eq i32 %stack_pos.0, 0
  br i1 %cmp.not, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %dec = add nsw i32 %stack_pos.0, -1
  %arrayidx = getelementptr inbounds i32, i32* %stack, i32 %dec
  %0 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx, i32 %lso.alloc2.0)
  %arrayidx1 = getelementptr inbounds i32, i32* %walk, i32 %walk_pos.0
  store i32 %0, i32* %arrayidx1, align 4, !tbaa !6
  %arrayidx2 = getelementptr inbounds i32, i32* %rows, i32 %0
  %1 = load i32, i32* %arrayidx2, align 4, !tbaa !6
  %add = add nsw i32 %0, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %rows, i32 %add
  %2 = load i32, i32* %arrayidx3, align 4, !tbaa !6
  br label %for.cond

for.cond:                                         ; preds = %if.end, %while.body
  %lso.alloc2.1 = phi i32 [ %lso.alloc2.0, %while.body ], [ %lso.alloc2.2, %if.end ]
  %lso.alloc1.1 = phi i32 [ %lso.alloc1.0, %while.body ], [ %lso.alloc1.2, %if.end ]
  %i.0 = phi i32 [ %1, %while.body ], [ %inc11, %if.end ]
  %stack_pos.1 = phi i32 [ %dec, %while.body ], [ %stack_pos.2, %if.end ]
  %cmp4 = icmp slt i32 %i.0, %2
  br i1 %cmp4, label %for.body, label %while.cond.loopexit

for.body:                                         ; preds = %for.cond
  %arrayidx5 = getelementptr inbounds i32, i32* %cols, i32 %i.0
  %3 = load i32, i32* %arrayidx5, align 4, !tbaa !6
  %arrayidx6 = getelementptr inbounds i32, i32* %visited, i32 %3
  %4 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx6, i32 %lso.alloc1.1)
  %tobool.not = icmp eq i32 %4, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx7 = getelementptr inbounds i32, i32* %stack, i32 %stack_pos.1
  %5 = call i32 (i32, i32*, ...) @cgra_store32(i32 %3, i32* %arrayidx7)
  %inc9 = add nsw i32 %stack_pos.1, 1
  %6 = call i32 (i32, i32*, ...) @cgra_store32(i32 1, i32* %arrayidx6, i32 %5)
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %lso.alloc2.2 = phi i32 [ %6, %if.then ], [ %lso.alloc2.1, %for.body ]
  %lso.alloc1.2 = phi i32 [ %6, %if.then ], [ %lso.alloc1.1, %for.body ]
  %stack_pos.2 = phi i32 [ %stack_pos.1, %for.body ], [ %inc9, %if.then ]
  %inc11 = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !10

while.end:                                        ; preds = %while.cond
  ret void
}
