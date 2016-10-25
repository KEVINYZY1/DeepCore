#ifndef __cgemm_h__
#define __cgemm_h__

#include"../../include/cuda/cuda_ctx.h"

void cgemm_flat_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int );
void cgemm_flat( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif