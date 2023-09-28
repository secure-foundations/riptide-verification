#include "stdint.h"

void test_1(int * restrict A, int len) {
    for (int i = 0; i < len; i++) {
        A[i] = i;
    }
}
