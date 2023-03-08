#include "stdint.h"

void sum(uint16_t * restrict A, uint16_t len) {
    for (uint32_t i = 0; i < len; i++) {
        A[i] += i;
    }
}
