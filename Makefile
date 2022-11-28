.PHONY: verify
verify:
	dafny /compile:0 /timeLimit:600 isa.dfy
