#ifndef __pooling_h__
#define __pooling_h__

#include"../../include/dc_argmask.h"
#include"../../include/cuda/cuda_ctx.h"

typedef struct poolingOp{
	cuda_kernel_t	kpooling[2];
	CUdeviceptr	d_max_id;
} poolingOp_t;

int	pooling_createOp( poolingOp_t*, const cuda_context_t*, int, int, int, int, int, int, int );
void 	pooling_launch( poolingOp_t*, CUdeviceptr, CUdeviceptr, int, CUstream );
void 	pooling_releaseOp( poolingOp_t* );

#endif
