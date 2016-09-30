#ifndef __gemv_h__
#define __gemv_h__

#include"../../include/dc_argmask.h"
#include"../../include/cuda/cuda_ctx.h"

void gemv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int );
void gemv( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif