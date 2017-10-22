#ifndef __idc_reduce_h__
#define __idc_reduce_h__

#include"../../include/cuda/cuda_ctx.h"

typedef struct idc_reduceOp{
    cuda_kernel_t kernel;
    CUdeviceptr   d_temp;
} idc_reduceOp_t;

int  idc_reduce_createOp( idc_reduceOp_t*, cuda_context_t*, uint32_t, int, int, int );
void idc_reduce_launch( idc_reduceOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void idc_reduce_releaseOp( idc_reduceOp_t* );

#endif