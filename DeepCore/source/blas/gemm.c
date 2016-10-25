#include"../../include/blas/gemm.h"

void gemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int prc, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
	static const char* knames[][6]=
	{
		{ "d_sgemm_128x32", "d_sgemm_128x32_bc", "d_sgemm_128x64", "d_sgemm_128x64_bc", "d_sgemm_128x128", "d_sgemm_128x128_bc" },
		{ "d_xgemm_128x32", "d_xgemm_128x32_bc", "d_xgemm_128x64", "d_xgemm_128x64_bc", "d_xgemm_128x128", "d_xgemm_128x128_bc" },
		{ "d_hgemm_256x32", "d_hgemm_256x32_bc", "d_hgemm_256x64", "d_hgemm_256x64_bc", "d_hgemm_256x128", "d_hgemm_256x128_bc" }
	};
	static const unsigned char size[]={127,127,255};
	int i=(cnc>32)+(((cnc&127)==0)|((cnc&127)>64));
	int tile_h=1<<(i+5);
	cuda_context_create_kernel( p_kernel, p_ctx, knames[prc][(i<<1)+((cnc&(tile_h-1))!=0)] );
	cuda_kernel_sao( p_kernel, AM_3P_7S );
	cuda_kernel_sbl( p_kernel, size[i]+1, 1 );
	cuda_kernel_sgl( p_kernel, (anr+p_kernel->block.x-1)/p_kernel->block.x, (cnc+tile_h-1)/tile_h );
	cuda_kernel_sep_i32( p_kernel, 4, anr );
	cuda_kernel_sep_i32( p_kernel, 5, bnr );
	cuda_kernel_sep_i32( p_kernel, 6, cnc );
	cuda_kernel_sep_i32( p_kernel, 7, lda );
	cuda_kernel_sep_i32( p_kernel, 8, ldb );
	cuda_kernel_sep_i32( p_kernel, 9, ldc );
}
void gemv_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int nx, int ny, int lda )
{
	const char* knames[][2]={{"d_sgemv", "d_sgemv_bc"},{"d_xgemv", "d_xgemv_bc"},{"d_hgemv", "d_hgemv_bc"}};
	unsigned int i=((nx&127)|(ny&31))!=0;
	unsigned int s=prc!=0;
	cuda_context_create_kernel( p, p_ctx, knames[prc][i] );
	cuda_kernel_sao( p, AM_3P_4S );
	cuda_kernel_sep_i32( p, 4, (nx+1)>>s );
	cuda_kernel_sep_i32( p, 5, ny	     );
	cuda_kernel_sep_i32( p, 6, lda>>s    );
}
void gemm( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float aph, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_c );
	cuda_kernel_sep_ptr( p_kernel, 1, d_a );
	cuda_kernel_sep_ptr( p_kernel, 2, d_b );
	cuda_kernel_sep_f32( p_kernel, 3, aph );
	cuda_kernel_launch( p_kernel, s );
}