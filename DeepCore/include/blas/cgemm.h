#ifndef __cgemm_h__
#define __cgemm_h__

#include"../../include/dc_argmask.h"
#include"../../include/cuda/cuda_ctx.h"

void cgemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int, int );
void cgemv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int );
void cgevv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int );
void cgemm( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif