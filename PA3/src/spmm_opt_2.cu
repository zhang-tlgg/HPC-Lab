#include "spmm_opt.h"
#include <vector>

#define WARP_SIZE 32
const int TaskSize = 256;

__global__ void spmm_kernel_opt_task_1(const Task *tasks, const int *idx, const float *val, const float *vin, float *vout, const int num_task, int feat_in) 
{     
    __shared__ int sm_idx[WARP_SIZE];
    __shared__ float sm_val[WARP_SIZE];
                        
    const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x >= num_task) return;

    const Task task = tasks[thread_id_x];
    const int row_id = task.row;
    const int col_id = blockIdx.y * WARP_SIZE + threadIdx.y;
    const int ptr_begin = task.ptr_begin;
    const int ptr_end = task.ptr_end;

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
    atomicAdd(&(vout[row_id * feat_in + col_id]), result);
}

__global__ void spmm_kernel_opt_task_2(const Task *tasks, const int *idx, const float *val, const float *vin, float *vout, const int num_task, int feat_in) 
{     
    __shared__ int sm_idx[WARP_SIZE];
    __shared__ float sm_val[WARP_SIZE];
                        
    const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x >= num_task) return;

    const Task task = tasks[thread_id_x];
    const int row_id = task.row;
    const int col_id = blockIdx.y * WARP_SIZE * 2 + threadIdx.y;
    const int ptr_begin = task.ptr_begin;
    const int ptr_end = task.ptr_end;

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
            result0 += sm_val[i] * __ldg(vin + v_idx);
            result1 += sm_val[i] *  __ldg(vin + v_idx + WARP_SIZE);
        }
    }
    int v_idx = row_id * feat_in + col_id;
    atomicAdd(&(vout[v_idx]), result0);
    atomicAdd(&(vout[v_idx + WARP_SIZE]), result1);
}

static inline int ceiling(int a, int b) {
    return (a + b - 1) / b;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    int *h_ptr = new int[num_v + 1];
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Task> tasks;
    for (int r = 0; r < num_v; ++r) {
        int begin = h_ptr[r];
        int end = h_ptr[r + 1];
        for (int ptr = begin; ptr < end; ptr += TaskSize) {
            Task task = {r, ptr, min(ptr + TaskSize, end)};
            tasks.push_back(task);
        }
    }
    delete[] h_ptr;
    num_task = tasks.size();
    std::random_shuffle(tasks.begin(), tasks.end());
    checkCudaErrors(cudaMalloc2((void **)&task_list, num_task * sizeof(Task)));
    checkCudaErrors(cudaMemcpy(task_list, tasks.data(), num_task * sizeof(Task), cudaMemcpyHostToDevice));
    block.x = 1;
    block.y = WARP_SIZE;
    grid.x = num_task;
    if (feat_in <= WARP_SIZE) {
        grid.y = ceiling(feat_in, WARP_SIZE);
    } else {
        grid.y = ceiling(feat_in, WARP_SIZE * 2);
    }
}

void SpMMOpt::run(float *vin, float *vout) {
    if (feat_in <= WARP_SIZE) {
        spmm_kernel_opt_task_1<<<grid, block>>>(task_list, d_idx, d_val, vin, vout, num_task, feat_in);
    } else {
        spmm_kernel_opt_task_2<<<grid, block>>>(task_list, d_idx, d_val, vin, vout, num_task, feat_in);
    }
}