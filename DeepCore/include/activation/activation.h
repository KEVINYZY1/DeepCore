#ifndef __activation_h__
#define __activation_h__

#include"../../include/cuda/cuda_ctx.h"

typedef struct activationOp{
	cuda_kernel_t kernel[2];
	short2        radix_ifunc;
} activationOp_t;

void activation_createOp( activationOp_t*, cuda_context_t*, int, int, int, int, int );
void activation_fprop( activationOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void activation_bprop( activationOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif