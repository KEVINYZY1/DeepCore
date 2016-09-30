#ifndef __perm_h__
#define __perm_h__

#include"../../include/dc_argmask.h"
#include"../../include/cuda/cuda_ctx.h"

void perm3d_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int );
void perm2d_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int );
void permute( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUstream );

#endif