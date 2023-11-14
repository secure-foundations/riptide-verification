void fft(
	int * restrict real,
	int * restrict imag,
	const int * restrict real_twiddle,
	const int * restrict imag_twiddle,
	int size, int stride, int step,
	int Ls, int theta,
	int strided_step, int Ls_stride) {

	for(int j = 0; j < Ls; j++) {
		int theta_idx = j + theta;
		int wr = real_twiddle[theta_idx];
		int wi = imag_twiddle[theta_idx];

		for(int k = j; k < size; k += step) {
			int re = real[j * stride + Ls_stride];
			int im = imag[j * stride + Ls_stride];
			int tr = wr * re - wi * im;
			int ti = wr * im + wi * re;

			re = real[j * stride];
			im = imag[j * stride];
			real[j * stride + Ls_stride] = re - tr;
			imag[j * stride + Ls_stride] = im - ti;
			real[j * stride] = re + tr;
			imag[j * stride] = im + ti;

			real += strided_step;
			imag += strided_step;
		}
	}
}
