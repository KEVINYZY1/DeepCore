#include"../../include/blas/gemv.h"

void gemv_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int nx, int ny, int lda )
{
	const char* knames[][2]={{"d_sgemv", "d_sgemv_bc"},{"d_xgemv", "d_xgemv_bc"}};
	unsigned int i=((nx&127)|(ny&31))!=0;
	unsigned int s=prc!=0;
	cuda_context_create_kernel( p, p_ctx, knames[prc][i] );
	cuda_kernel_sao( p, AM_3P_4S );
	cuda_kernel_sep_i32( p, 4, (nx+1)>>s );
	cuda_kernel_sep_i32( p, 5, ny	     );
	cuda_kernel_sep_i32( p, 6, lda>>s    );
}
void gemv( cuda_kernel_t* p, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float aph, CUstream s )
{
	cuda_kernel_sep_ptr( p, 0, d_c );
	cuda_kernel_sep_ptr( p, 1, d_a );
	cuda_kernel_sep_ptr( p, 2, d_b );
	cuda_kernel_sep_f32( p, 3, aph );
	cuda_kernel_launch( p, s );
}
