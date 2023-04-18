#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    // Your code here
    __m256 am, bm, cm;
    for (int i = 0; i < n; i += 8){
        am = _mm256_load_ps(&a[i]);
        bm = _mm256_load_ps(&b[i]);
        cm = _mm256_add_ps(am, bm);
        _mm256_store_ps(&c[i], cm);
    }
}