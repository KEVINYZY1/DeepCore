#ifndef __conv_h__
#define __conv_h__

#include"../../include/cuda/cuda_ctx.h"

typedef struct convOp{
	cuda_kernel_t kernel;
	CUdeviceptr   d_slider;
	CUdeviceptr   d_slider_cmem; 
	int           slider_size;
} convOp_t;

int  conv_createOp( convOp_t*, unsigned int*, const cuda_context_t*, unsigned int, int, int, int, int, int, int );
void conv( convOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, const float*, CUstream );
void conv_releaseOp( convOp_t* );

#endif
