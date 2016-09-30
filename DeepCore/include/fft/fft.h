#ifndef __fft_h__
#define __fft_h__

#include"../../include/cuda/cuda_ctx.h"
#include"../../include/dc_argmask.h"

void create_fft_kernel_r2c( cuda_kernel_t*, const cuda_context_t*, unsigned int, unsigned int );
void create_fft_kernel_c2r( cuda_kernel_t*, const cuda_context_t*, unsigned int, unsigned int );
void create_cellfft_kernel_r2c( cuda_kernel_t*, const cuda_context_t*, unsigned int, unsigned int );
void create_cellfft_kernel_c2r( cuda_kernel_t*, const cuda_context_t*, unsigned int, unsigned int );

__forceinline CUresult fft2d( cuda_kernel_t* p_kernel, CUdeviceptr d_dst, CUdeviceptr d_src, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_dst  );
	cuda_kernel_sep_ptr( p_kernel, 1, d_src  );
	return cuda_kernel_launch( p_kernel, s );
}

#endif