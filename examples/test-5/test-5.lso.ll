define void @test(i32* %A, i32* %B, i32 %len) {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.cleanup, %entry
  %lso.alloc1.0 = phi i32 [ 0, %entry ], [ %lso.alloc1.1.lcssa, %outer.cleanup ]
  %i = phi i32 [ 0, %entry ], [ %inc.i, %outer.cleanup ]
  %outer.cond = icmp slt i32 %i, %len
  br i1 %outer.cond, label %outer.body, label %end

outer.body:                                       ; preds = %outer.header
  br label %inner.header

inner.header:                                     ; preds = %inner.body, %outer.body
  %lso.alloc1.1 = phi i32 [ %lso.alloc1.0, %outer.body ], [ %1, %inner.body ]
  %j = phi i32 [ 0, %outer.body ], [ %inc.j, %inner.body ]
  %inner.cond = icmp slt i32 %j, %len
  br i1 %inner.cond, label %inner.body, label %outer.cleanup

inner.body:                                       ; preds = %inner.header
  %A.i = getelementptr inbounds i32, i32* %A, i32 %i
  %B.j = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = call i32 (i32*, ...) @cgra_load32(i32* %B.j, i32 %lso.alloc1.1)
  %1 = call i32 (i32, i32*, ...) @cgra_store32(i32 %0, i32* %A.i)
  %inc.j = add i32 %j, 1
  br label %inner.header

outer.cleanup:                                    ; preds = %inner.header
  %lso.alloc1.1.lcssa = phi i32 [ %lso.alloc1.1, %inner.header ]
  %inc.i = add i32 %i, 1
  br label %outer.header

end:                                              ; preds = %outer.header
  ret void
}
