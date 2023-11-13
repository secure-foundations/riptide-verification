define void @constant_pe(i32* %a, i32 %b, i32 %len) {
entry:
    br label %header
        
header:
    %i = phi i32 [ 0, %entry ], [ %i.0, %body ]
    %loop.cond = icmp slt i32 %i, %len
    br i1 %loop.cond, label %body, label %end

body:
    %b.0 = add i32 %b, 1
    %elem = getelementptr i32, i32* %a, i32 %i
    store i32 %b.0, i32* %elem
    %i.0 = add i32 %i, 1
    br label %header

end:
    ret void
}
