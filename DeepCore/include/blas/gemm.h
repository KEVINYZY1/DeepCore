#ifndef __gemm_h__
#define __gemm_h__

#include"../../include/cuda/cuda_ctx.h"

void gemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int );
void gemm( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif