void fft_ns0(
    int * restrict src_real_ptr,
    int * restrict src_imag_ptr,
    int * restrict dst_real_ptr,
    int * restrict dst_imag_ptr, 
	int size, int stride, 
	int i_2, int i_1, int mask
) {

	for(int i = 0; i < size; i++) {
		int i2 = i + i_2;
		int i1 = i2 & i_1;
		int m = i & mask;
		int i1_m = i1 + m; 
		int idx = i * stride;
		dst_real_ptr[i1_m] = src_real_ptr[idx];
		dst_imag_ptr[i1_m] = src_imag_ptr[idx];
	}
}
