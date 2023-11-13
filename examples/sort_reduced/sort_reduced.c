void sort_reduced(int * restrict A, int * restrict Z, int size) {
	// int count = even_count;
	for(int i = 0; i < 32; i++) {
		if (i & 0x1) {
			for(int j = 0; j < size; j++) {
				A[j] = 0;
			}
		} else {
			for(int j = 0; j < size; j++) {
				Z[j] = 0;
			}
		}
	}
}
