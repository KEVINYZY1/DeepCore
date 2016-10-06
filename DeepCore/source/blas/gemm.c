#include"../../include/blas/gemm.h"

void gemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int prc, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
	static const char* knames[][]=
	{
		{ "d_sgemm_128x32", "d_sgemm_128x32_bc", "d_sgemm_128x64", "d_sgemm_128x64_bc", "d_sgemm_128x128", "d_sgemm_128x128_bc" },
		{ "d_xgemm_128x32", "d_xgemm_128x32_bc", "d_xgemm_128x64", "d_xgemm_128x64_bc", "d_xgemm_128x128", "d_xgemm_128x128_bc" },
		{ "d_hgemm_128x32", "d_hgemm_128x32_bc", "d_hgemm_128x64", "d_hgemm_128x64_bc", "d_hgemm_128x128", "d_hgemm_128x128_bc" }
	};
}
void gemm( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float aph, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_c );
	cuda_kernel_sep_ptr( p_kernel, 1, d_a );
	cuda_kernel_sep_ptr( p_kernel, 2, d_b );
	cuda_kernel_sep_f32( p_kernel, 3, aph );
	cuda_kernel_launch( p_kernel, s );
}