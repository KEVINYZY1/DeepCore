#ifndef __bnorm_h__
#define __bnorm_h__

#include"../../include/cuda/cuda_ctx.h"
#include"../../include/idc_string.h"

void idc_bnorm_createOp( cuda_kernel_t*, const cuda_context_t*, uint32_t mask, int size, int pitch, int nc );
void idc_bnorm( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUstream );
void idc_bnorm_grad( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUstream );

#endif