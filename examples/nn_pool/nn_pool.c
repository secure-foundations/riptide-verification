#include "stdint.h"
#include "stddef.h"
#include "limits.h"

void nn_pool(int * src, int * dest,
	int input_rows_bump, int input_cols,
	int output_size, int output_cols, int pool_size) {

	int * dest_ptr = dest;
	int * src_ptr = src;
	int col = 0;

	for(int i = 0; i < output_size; i++) {

		int * src_pool_ptr = src_ptr;
		int w = SHRT_MIN;

		for(int j = 0; j < pool_size; j++) {

			for(int k = 0; k < pool_size; k++) {
				int t = src_pool_ptr[k];
				if(t > w) w = t;
			}

			src_pool_ptr += input_cols;
		}

		dest_ptr[i] = w;
		src_ptr += pool_size;
		col++;
		if(col == output_cols) {
			col = 0;
			src_ptr += input_rows_bump;
		}

	}

}