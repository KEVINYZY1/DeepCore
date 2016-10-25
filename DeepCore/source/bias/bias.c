#include"../../include/bias/bias.h"

size_t bias_createOp( biasOp_t* Op, const cuda_context_t* p_ctx, int prc, int size, int nc )
{
	int align, radix, i, n, pitch, gdx, gdy;
	static const char* symbols[][5]=
	{
		{
			"d_sbias_update_032",
			"d_sbias_update_064",
			"d_sbias_update_128",
			"d_sbias_update_256",
			"d_sbias_update"
		},
		{
			"d_xbias_update_032",
			"d_xbias_update_064",
			"d_xbias_update_128",
			"d_xbias_update_256",
			"d_xbias_update"
		}
	};
	static const struct{ unsigned char x, y; } block_shape[4]={{31,8},{63,1},{127,1},{255,1}};
	
	align=prc?(BASE_PITCH/2):(BASE_PITCH/4);
	radix=prc?8:4;
	n=(size+radix-1)/radix;
	pitch=AFFIS(size,align)/radix;
	radix=p_ctx->n_sm<<3;
	i=(n>=64)+(n>=128)+(n>=256)+(nc>radix);
	gdx=nc;
	if(i==4){
		gdx=(n+255)>>8;
		gdx=gdx>radix?radix:gdx;
	}
	gdy=i<4?1:nc;
	cuda_context_create_kernel( &Op->kupdate, p_ctx, symbols[prc][i] );
	cuda_kernel_sao( &Op->kupdate, (i<4)?AM_2P_3S:AM_4P_3S );	
	cuda_kernel_sbl( &Op->kupdate, block_shape[i].x+1, block_shape[i].y );
	cuda_kernel_sgl( &Op->kupdate, gdx, gdy );
	i=i<4?2:4;
	cuda_kernel_sep_i32( &Op->kupdate, i+0, n     );
	cuda_kernel_sep_i32( &Op->kupdate, i+1, pitch );
	n=0;
	if(gdy>1){ n=(AFFIS(nc,32)+gdx*nc)*sizeof(int);	}
	return (size_t)n;
}
void bias_update( biasOp_t* Op, CUdeviceptr d_temp, CUdeviceptr d_b, CUdeviceptr d_a, float eta, CUstream s )
{
	cuda_kernel_t* p=&Op->kupdate;
	int i=(p->gdy>1)?2:0;
	if(p->gdy>1){
		cuMemsetD8Async( d_temp, 0, p->gdy*sizeof(int), s );
		cuda_kernel_sep_ptr( p, 0, d_temp );
		cuda_kernel_sep_ptr( p, 1, d_temp+AFFIS(p->gdy,32) );
	}
	cuda_kernel_sep_ptr( p, i+0, d_b );
	cuda_kernel_sep_ptr( p, i+1, d_a );
	cuda_kernel_sep_f32( p, i+4, eta );
	cuda_kernel_launch( p, s );
}
