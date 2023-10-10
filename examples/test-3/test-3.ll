define void @test(i32* %A, i32* %B, i32 %len) {
entry:
  br label %outer.header

outer.header:
  %i = phi i32 [ 0, %entry ], [ %inc.i, %outer.cleanup ]
  %outer.cond = icmp slt i32 %i, %len
  br i1 %outer.cond, label %outer.body, label %end

outer.body:
  br label %inner.header

inner.header:
  %j = phi i32 [ 0, %outer.body ], [ %inc.j, %inner.body ]
  %inner.cond = icmp slt i32 %j, %len
  br i1 %inner.cond, label %inner.body, label %outer.cleanup

inner.body:
  %A.i = getelementptr inbounds i32, i32* %A, i32 %i
  %B.j = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %B.j
  store i32 %0, i32* %A.i
  %inc.j = add i32 %j, 1
  br label %inner.header

outer.cleanup:
  %inc.i = add i32 %i, 1
  br label %outer.header

end:
  ret void
}
