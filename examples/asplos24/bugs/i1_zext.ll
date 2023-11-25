define void @i1_zext(i32* %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %b, 0
  %cmp.1 = xor i1 %cmp, true
  %conv = zext i1 %cmp.1 to i32
  store i32 %conv, i32* %a
  ret void
}
