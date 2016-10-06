#include"../../include/conv/conv.h"
#include"../../include/blas/gemm.h"
#include"../../include/blas/gemv.h"

static void __generate_slider( unsigned int* p_slider, int m, int pm, int n, int pn, int ds, int fs, int inc, int st, int enb )
{
	int u, v, c, i;
	for( c=0; c<inc; ++c )
	{
		for( v=0; v<fs; ++v ){
			for( u=0; u<fs; ++u ){
				*p_slider++=c*pm+(v*st*ds+u*st)*enb;
			}
		}
	}
	for( i=inc*pm-(m<pm)*enb, pn-=n; pn>0; --pn ){ *p_slider++=i; }
}
int conv_createOp( convOp_t* Op, unsigned int* p_temp, const cuda_context_t* p_ctx, int prc, int ds, int fs, int bat, int inc, int onc, int st )
{
	int enb, os, anr, bnr, cnr, lda, ldb, ldc;
	static const char* knames[]={ "d_sconv_64x16", "d_sconv_64x16_bc", "d_sconv_128x32", "d_sconv_128x32_bc", "d_sconv_128x64", "d_sconv_128x64_bc", "d_sconv_128x128", "d_sconv_128x128_bc" };
	cuda_kernel_t* p_kernel=&Op->kernel;

	enb=prc?2:4;
	anr=bat*ds*ds;
	bnr=inc*fs*fs;
	os=(ds-fs)/st+1;
	lda=AFFIS(anr*enb,BASE_PITCH);
	ldb=AFFIS(bnr*enb,BASE_PITCH);
	ldc=AFFIS(cnr*enb,BASE_PITCH);
	Op->d_slider=0;
	if((ds!=fs)&(fs>1))
	{
		int pn, vs, i, s, tile_y, is_bc; 
		if(cuMemAlloc( &Op->d_slider, sizeof(int)*(pn=AFFIS(bnr,8)) )!=CUDA_SUCCESS)
		    return ERROR_OUT_OF_DEVICE_MEMORY;
		__generate_slider( p_temp, anr*enb, lda, bnr, pn, ds, fs, st, inc, enb );
		cuMemcpyHtoD( Op->d_slider, p_temp, sizeof(int)*pn );	
		vs=AFFIS(anr,2);
		i=(onc>16)+(onc>32)+(((onc&127)==0)|((onc&127)>64));
		s=4+i;
		tile_y=1<<s;
		is_bc=(onc&(tile_y-1))!=0;
		cuda_context_create_kernel( p_kernel, p_ctx, knames[(i<<1)+is_bc] );
		cuda_kernel_sao( p_kernel, AM_4P_9S );
		cuda_kernel_sbl( p_kernel, i?(i<3?128:256):64, 1 );
		cuda_kernel_sgl( p_kernel, (vs+p_kernel->block.x-1)/p_kernel->block.x, (onc+tile_y-1)>>s );
		cuda_kernel_sep_i32( p_kernel, 5, ldc	);
		cuda_kernel_sep_i32( p_kernel, 6, ldb	);
		cuda_kernel_sep_i32( p_kernel, 7, os	);
		cuda_kernel_sep_i32( p_kernel, 8, os	);
		cuda_kernel_sep_i32( p_kernel, 9, ds	);
		cuda_kernel_sep_i32( p_kernel,10, ds	);
		cuda_kernel_sep_i32( p_kernel,11, pn	);
		cuda_kernel_sep_i32( p_kernel,12, vs	);
		cuda_kernel_sep_i32( p_kernel,13, onc	);
	} else
	if(fs==1)
	{
	}
	return SUCCESS;
}
void conv( convOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, float alpha, float* p_beta, CUstream s )
{
	cuda_kernel_t* p=&Op->kernel;
	cuda_kernel_sep_ptr( p, 0, d_c );
	cuda_kernel_sep_ptr( p, 1, d_a );
	cuda_kernel_sep_ptr( p, 2, d_b );
	if(Op->d_slider!=0){
	    cuda_kernel_sep_ptr( p, 3, Op->d_slider );
	}
	cuda_kernel_sep_f32( p, 3+(Op->d_slider!=0), alpha );
	cuda_kernel_launch( p, s );
}
void conv_releaseOp( convOp_t* Op )
{
	if(Op->d_slider!=0){
	    cuMemFree(Op->d_slider);
	    Op->d_slider=0;
	}
}
