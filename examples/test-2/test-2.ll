define void @test(i32* %A, i32* %B, i32 %len) {
entry:
  %b = load i32, i32* %B
  br label %header

header:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %body ]
  %cond = icmp slt i32 %i.0, %len
  br i1 %cond, label %body, label %end

body:
  %arrayidx = getelementptr i32, i32* %A, i32 %i.0
  store i32 %b, i32* %arrayidx
  %inc = add i32 %i.0, 1
  br label %header

end:
  ret void
}
