#ifndef __bias_h__
#define __bias_h__

#include"../../include/cuda/cuda_ctx.h"

typedef struct biasOp{
	cuda_kernel_t kupdate;
} biasOp_t;

size_t bias_createOp( biasOp_t*, const cuda_context_t*, int, int, int );
void   bias_update( biasOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif