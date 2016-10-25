#ifndef __fft_h__
#define __fft_h__

#include"../../include/cuda/cuda_ctx.h"

void create_fft_kernel_r2c( cuda_kernel_t*, const cuda_context_t*, int, int );
void create_fft_kernel_c2r( cuda_kernel_t*, const cuda_context_t*, int, int );
void create_cellfft_kernel_r2c( cuda_kernel_t*, const cuda_context_t*, int, int );
void create_cellfft_kernel_c2r( cuda_kernel_t*, const cuda_context_t*, int, int );

__forceinline CUresult fft2d_r2c( cuda_kernel_t* p_kernel, CUdeviceptr d_dst, CUdeviceptr d_src, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_dst );
	cuda_kernel_sep_ptr( p_kernel, 1, d_src );
	return cuda_kernel_launch( p_kernel, s );
}
__forceinline CUresult fft2d_c2r( cuda_kernel_t* p_kernel, CUdeviceptr d_dst, CUdeviceptr d_src, CUdeviceptr d_bias_or_atv, const float* alpha, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_dst );
	cuda_kernel_sep_ptr( p_kernel, 1, d_src );
	if(d_bias_or_atv!=0){
		cuda_kernel_sep_ptr( p_kernel, 3, d_bias_or_atv );
	}
	if(alpha!=NULL){
		cuda_kernel_sep_f32( p_kernel, 3+(d_bias_or_atv!=0), alpha[0] );
	}
	return cuda_kernel_launch( p_kernel, s );
}

#endif
