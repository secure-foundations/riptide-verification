#include "stdint.h"

void disjoint(uint16_t * restrict array1,
              uint16_t * restrict array2,
              uint16_t len) {
    for (uint32_t i = 0; i < len; i++) {
        array1[i] = i;
        array2[i] = i;
    }
}
