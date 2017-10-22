#ifndef __conv_h__
#define __conv_h__

#include"../../include/blas/gemm.h"

typedef struct idc_convOp{
    cuda_kernel_t kernel;
    CUdeviceptr   d_indices;
    CUdeviceptr   d_indices_cmem;
    int           indices_nb;
} idc_convOp_t;

int  idc_conv_createOp( idc_convOp_t*, uint32_t*, const cuda_context_t*, int, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int );
void idc_conv( idc_convOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void idc_conv_relu( idc_convOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, float, CUstream );
void idc_conv_releaseOp( idc_convOp_t* );

#endif
