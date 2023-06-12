#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

struct Task {
  int row;
  int ptr_begin;
  int ptr_end;
};

class SpMMOpt : public SpMM {
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (this->task_list) checkCudaErrors(cudaFree(this->task_list));
    }
    virtual void preprocess(float *vin, float *vout);
    virtual void run(float *vin, float *vout);

private:
    int num_task;
    Task *task_list;
};
#endif