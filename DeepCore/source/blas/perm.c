#include"../../include/blas/perm.h"

void perm3d_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int idx, int nx, int ny, int nz, int lda, int ldb )
{
	static const char* knames[2][3]=
	{
		{ "d_perm_xzy_b64", "d_perm_xyz_b64", "d_perm_yxz_b64" },
		{ "d_perm_xzy_b32", "d_perm_xyz_b32", "d_perm_yxz_b32" }
	};
	int n[][3]={{nx,ny,nz},{nx,nz,ny},{nx,nz,ny}};
	int gdx=((n[idx][0]+31)>>5)*((n[idx][1]+31)>>5);
	int gdy=n[idx][2];
	cuda_context_create_kernel( p, p_ctx, knames[prc][idx] );
	cuda_kernel_sao( p, AM_2P_5S );	
	cuda_kernel_sgl( p, gdx, gdy );
	cuda_kernel_sbl( p, 1024, 1 );	
	cuda_kernel_sep_i32( p, 2, nx	);
	cuda_kernel_sep_i32( p, 3, ny	);
	cuda_kernel_sep_i32( p, 4, nz	);
	cuda_kernel_sep_i32( p, 5, lda	);
	cuda_kernel_sep_i32( p, 6, ldb	);
}
void perm2d_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int nx, int ny, int lda, int ldb )
{
	cuda_context_create_kernel( p, p_ctx, prc?"d_perm2d_b32":"d_perm2d_b64" );
	cuda_kernel_sao( p, AM_2P_4S );	
	cuda_kernel_sgl( p, (nx+31)>>5, (ny+31)>>5 );
	cuda_kernel_sbl( p, 32, 32 );	
	cuda_kernel_sep_i32( p, 2, nx	);
	cuda_kernel_sep_i32( p, 3, ny	);
	cuda_kernel_sep_i32( p, 4, lda	);
	cuda_kernel_sep_i32( p, 5, ldb	);
}
void permute( cuda_kernel_t* p, CUdeviceptr d_b, CUdeviceptr d_a, CUstream s )
{
	cuda_kernel_sep_ptr( p, 0, d_b );
	cuda_kernel_sep_ptr( p, 1, d_a );
	cuda_kernel_launch( p, s );
}