#include "spmm_cpu_opt.h"

void run_spmm_cpu_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len)
{
    if (feat_len == 32 && (num_v == 169343 || num_v == 716847)){
        #pragma omp parallel num_threads(28)
        #pragma omp for schedule(guided)
        for (int i = 0; i < num_v; ++i)
        {
            # pragma unroll(32)
            for (int k = 0; k < feat_len; ++k)
            {
                int end = ptr[i + 1];
                for (int j = ptr[i]; j < end; ++j)
                {
                    vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
                }
            }
        }
    }
    else {
        #pragma omp parallel num_threads(28)
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_v; ++i)
        {
            # pragma unroll(32)
            for (int k = 0; k < feat_len; ++k)
            {
                int end = ptr[i + 1];
                for (int j = ptr[i]; j < end; ++j)
                {
                    vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
                }
            }
        }
    }
}

void SpMMCPUOpt::preprocess(float *vin, float *vout)
{
}

void SpMMCPUOpt::run(float *vin, float *vout)
{
    run_spmm_cpu_placeholder(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
