#include "stdint.h"
#include "stddef.h"
#include "limits.h"

void nn_fc(int * weight, int * src, int * dest,
	int rows, int cols, int shift) {

	int * weight_ptr = weight;
	int * dest_ptr = dest;

	for(int i = 0; i < rows; i++) {
		int w = 0;
		int * src_ptr = src;

		for(int j = 0; j < cols; j++) {
			int s = src_ptr[j];
			int f = weight_ptr[j];
			w += s * f;
		}

		w >>= shift;
		if(w < SHRT_MIN) w = SHRT_MIN;
		if(w > SHRT_MAX) w = SHRT_MAX;

		dest_ptr[i] = w;
		weight_ptr += cols;
	}

}