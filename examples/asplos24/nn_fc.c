#include "stdint.h"
#include "stddef.h"
#include "limits.h"

void nn_fc(int * restrict weight, int * restrict src, int * restrict dest,
	int rows, int cols, int shift) {

	for(int i = 0; i < rows; i++) {
		int w = 0;

		for(int j = 0; j < cols; j++) {
			int s = src[j];
			int f = weight[i * cols + j];
			w += s * f;
		}

		w >>= shift;
		if(w < SHRT_MIN) w = SHRT_MIN;
		if(w > SHRT_MAX) w = SHRT_MAX;

		dest[i] = w;
	}

}