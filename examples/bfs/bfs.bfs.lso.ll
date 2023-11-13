; Function Attrs: minsize nofree norecurse nounwind optsize ssp
define dso_local void @bfs(i32* noalias nocapture readonly %rows, i32* noalias nocapture readonly %cols, i32 %count, i32* nocapture %queue, i32* nocapture %visited, i32* nocapture %walk) local_unnamed_addr #0 {
entry:
  br label %while.cond

while.cond.loopexit:                              ; preds = %for.cond
  %lso.alloc3.1.lcssa = phi i32 [ %lso.alloc3.1, %for.cond ]
  %lso.alloc2.1.lcssa = phi i32 [ %lso.alloc2.1, %for.cond ]
  %queue_back.1.lcssa = phi i32 [ %queue_back.1, %for.cond ]
  %inc2 = add nuw nsw i32 %walk_pos.0, 1
  br label %while.cond, !llvm.loop !4

while.cond:                                       ; preds = %while.cond.loopexit, %entry
  %lso.alloc3.0 = phi i32 [ 0, %entry ], [ %lso.alloc3.1.lcssa, %while.cond.loopexit ]
  %lso.alloc2.0 = phi i32 [ 0, %entry ], [ %lso.alloc2.1.lcssa, %while.cond.loopexit ]
  %walk_pos.0 = phi i32 [ 0, %entry ], [ %inc2, %while.cond.loopexit ]
  %queue_back.0 = phi i32 [ 1, %entry ], [ %queue_back.1.lcssa, %while.cond.loopexit ]
  %cmp.not = icmp eq i32 %walk_pos.0, %queue_back.0
  br i1 %cmp.not, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %arrayidx = getelementptr inbounds i32, i32* %queue, i32 %walk_pos.0
  %0 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx, i32 undef, i32 %lso.alloc3.0)
  %arrayidx1 = getelementptr inbounds i32, i32* %walk, i32 %walk_pos.0
  %1 = call i32 (i32, i32*, ...) @cgra_store32(i32 %0, i32* %arrayidx1, i32 undef, i32 %lso.alloc2.0)
  %arrayidx3 = getelementptr inbounds i32, i32* %rows, i32 %0
  %2 = load i32, i32* %arrayidx3, align 4, !tbaa !6
  %add = add nsw i32 %0, 1
  %arrayidx4 = getelementptr inbounds i32, i32* %rows, i32 %add
  %3 = load i32, i32* %arrayidx4, align 4, !tbaa !6
  br label %for.cond

for.cond:                                         ; preds = %if.end, %while.body
  %lso.alloc3.1 = phi i32 [ %lso.alloc3.0, %while.body ], [ %lso.alloc3.2, %if.end ]
  %lso.alloc2.1 = phi i32 [ %lso.alloc2.0, %while.body ], [ %lso.alloc2.2, %if.end ]
  %i.0 = phi i32 [ %2, %while.body ], [ %inc12, %if.end ]
  %queue_back.1 = phi i32 [ %queue_back.0, %while.body ], [ %queue_back.2, %if.end ]
  %cmp5 = icmp slt i32 %i.0, %3
  br i1 %cmp5, label %for.body, label %while.cond.loopexit

for.body:                                         ; preds = %for.cond
  %arrayidx6 = getelementptr inbounds i32, i32* %cols, i32 %i.0
  %4 = load i32, i32* %arrayidx6, align 4, !tbaa !6
  %arrayidx7 = getelementptr inbounds i32, i32* %visited, i32 %4
  %5 = call i32 (i32*, ...) @cgra_load32(i32* %arrayidx7, i32 %1)
  %tobool.not = icmp eq i32 %5, 0
  br i1 %tobool.not, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx8 = getelementptr inbounds i32, i32* %queue, i32 %queue_back.1
  %6 = call i32 (i32, i32*, ...) @cgra_store32(i32 %4, i32* %arrayidx8)
  %inc10 = add nsw i32 %queue_back.1, 1
  %7 = call i32 (i32, i32*, ...) @cgra_store32(i32 1, i32* %arrayidx7, i32 %6)
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %lso.alloc3.2 = phi i32 [ %7, %if.then ], [ %5, %for.body ]
  %lso.alloc2.2 = phi i32 [ %7, %if.then ], [ %5, %for.body ]
  %queue_back.2 = phi i32 [ %queue_back.1, %for.body ], [ %inc10, %if.then ]
  %inc12 = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !10

while.end:                                        ; preds = %while.cond
  ret void
}
