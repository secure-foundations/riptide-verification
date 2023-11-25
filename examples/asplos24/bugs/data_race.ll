define void @data_race(i32* noalias nocapture %A, i32 %n) {
entry:
  br label %outer.header

outer.header:
  %i.0 = phi i32 [ 0, %entry ], [ %inc8, %outer.cleanup ]
  %outer.cond = icmp slt i32 %i.0, %n
  br i1 %outer.cond, label %outer.body, label %end

outer.body:
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.0
  store i32 %i.0, i32* %arrayidx
  br label %inner.header

inner.header:
  %j.0 = phi i32 [ 1, %outer.body ], [ %inc, %inner.body ]
  %inner.cond = icmp slt i32 %j.0, %n
  br i1 %inner.cond, label %inner.body, label %outer.cleanup

inner.body:
  %sub = add nsw i32 %j.0, -1
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i32 %sub
  %0 = load i32, i32* %arrayidx5
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %j.0
  store i32 %0, i32* %arrayidx6
  %inc = add nuw i32 %j.0, 1
  br label %inner.header

outer.cleanup:
  %inc8 = add nuw i32 %i.0, 1
  br label %outer.header

end:
  ret void
}
