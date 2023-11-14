#include "stdint.h"
#include "stddef.h"
#include "limits.h"

void nn_vadd(int * restrict weight, int * restrict src, int * restrict dest, int size) {
	for(int i = 0; i < size; i++) {
		dest[i] = src[i] + weight[i];
	}
}