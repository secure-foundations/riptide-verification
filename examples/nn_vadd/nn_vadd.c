#include "stdint.h"
#include "stddef.h"
#include "limits.h"

void nn_vadd(int * weight, int * src, int * dest,
	int size) {

	int * src_ptr = src;
	int * weight_ptr = weight;
	int * dest_ptr = dest;

	for(int i = 0; i < size; i++) {
		*dest_ptr++ = *src_ptr++ + *weight_ptr++;
	}

}