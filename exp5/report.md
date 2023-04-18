# MPI 异步通信小作业 实验报告

张天乐 2018011038

## 运行时间

| 版本        | 时间      |
| --------- | ------- |
| baseline  | 4779 us |
| auto simd | 563 us  |
| intrinsic | 541 us  |

## 代码

```cpp
#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    __m256 am, bm, cm;
    for (int i = 0; i < n; i += 8){
        am = _mm256_load_ps(&a[i]);
        bm = _mm256_load_ps(&b[i]);
        cm = _mm256_add_ps(am, bm);
        _mm256_store_ps(&c[i], cm);
    }
}
```
