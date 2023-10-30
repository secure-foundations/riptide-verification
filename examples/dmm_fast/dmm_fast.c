void dmm_fast(
    const int * restrict A, 
	const int * restrict B,
    const int * restrict B_offset2,
	int * restrict Z, 
	int m, int n, int p, int p_half) {

	const int * filter_ptr = A;
	int * dest_ptr = Z;
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < p_half; j++) {
			int w = 0;
			int x = 0;
			int idx = ((j >> 1) << 2) + (j & 0x1);

			int src_idx = idx;
			for(int k = 0; k < n; k++) {
				int f = filter_ptr[k];
				w += f * B[src_idx];	
				x += f * B_offset2[src_idx];
				src_idx += p;
			}

			dest_ptr[idx] = w;
			dest_ptr[idx + 2] = x;
		}
		filter_ptr += n;
		dest_ptr += p;
	}
}
