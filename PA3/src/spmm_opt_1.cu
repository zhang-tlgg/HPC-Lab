#include "spmm_opt.h"
#include <vector>

#define WARP_SIZE 32

__global__ void spmm_kernel_opt_1(int *ptr, int *idx, float *val,
    float *vin, float *vout, int num_v, int feat_in) {

    __shared__ int sm_idx[WARP_SIZE];
    __shared__ float sm_val[WARP_SIZE];

    int row_id = blockIdx.x;
    int col_id = blockIdx.y * WARP_SIZE + threadIdx.y;

    if (row_id >= num_v) return;

    int ptr_begin = __ldg(ptr + row_id);
    int ptr_end = __ldg(ptr + row_id + 1);

    float result = 0.0f;

    for (int ptr = ptr_begin; ptr < ptr_end; ptr += WARP_SIZE) {
        int thr_ptr = ptr + threadIdx.y;
        if (thr_ptr < ptr_end) {
            sm_idx[threadIdx.y] = idx[thr_ptr];
            sm_val[threadIdx.y] = val[thr_ptr];
        }
        __syncthreads();
        int sm_end = min(WARP_SIZE, ptr_end - ptr);
        for (int i = 0; i < sm_end; i++) {
            int v_idx = sm_idx[i] * feat_in + col_id;
            result += sm_val[i] * __ldg(vin + v_idx);
        }
    }
    vout[row_id * feat_in + col_id] = result;
}

__global__ void spmm_kernel_opt_2(int *ptr, int *idx, float *val,
    float *vin, float *vout, int num_v, int feat_in) {

    __shared__ int sm_idx[WARP_SIZE];
    __shared__ float sm_val[WARP_SIZE];

    int row_id = blockIdx.x;
    int col_id = blockIdx.y * WARP_SIZE * 2 + threadIdx.y;

    if (row_id >= num_v) return;

    int ptr_begin = __ldg(ptr + row_id);
    int ptr_end = __ldg(ptr + row_id + 1);

    float result0 = 0.0f;
    float result1 = 0.0f;

    for (int ptr = ptr_begin; ptr < ptr_end; ptr += WARP_SIZE) {
        int thr_ptr = ptr + threadIdx.y;
        if (thr_ptr < ptr_end) {
            sm_idx[threadIdx.y] = idx[thr_ptr];
            sm_val[threadIdx.y] = val[thr_ptr];
        }
        __syncthreads();
        int sm_end = min(WARP_SIZE, ptr_end - ptr);
        for (int i = 0; i < sm_end; i++) {
            int v_idx = sm_idx[i] * feat_in + col_id;
            result0 += sm_val[i] *  __ldg(vin + v_idx);
            result1 += sm_val[i] *  __ldg(vin + v_idx + WARP_SIZE);
        }
    }
    int v_idx = row_id * feat_in + col_id;
    vout[v_idx] = result0;
    vout[v_idx + WARP_SIZE] = result1;
}

static inline int ceiling(int a, int b) {
    return (a + b - 1) / b;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
        block.x = 1;
        block.y = WARP_SIZE;
        grid.x = num_v;

        if (feat_in <= WARP_SIZE) {
            grid.y = ceiling(feat_in, WARP_SIZE);
        } else {
            grid.y = ceiling(feat_in, WARP_SIZE * 2);
        }
}

void SpMMOpt::run(float *vin, float *vout) {
    if (feat_in <= WARP_SIZE) {
        spmm_kernel_opt_1<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    } else {
        spmm_kernel_opt_2<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }
}