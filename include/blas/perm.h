#ifndef __perm_h__
#define __perm_h__

#include"../../include/cuda/cuda_ctx.h"

void idc_perm3d_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int );
void idc_perm2d_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int );
void idc_permute( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUstream );

#endif