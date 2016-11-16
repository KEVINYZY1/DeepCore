#include"../../include/conv/conv.h"
#include"../../include/blas/gemm.h"

static void generate_slider_conv( unsigned int* p_slider, int m, int pm, int n, int pn, int ds, int fs, int nc, int st, int enb )
{
	int u, v, c, i;
	for( c=0; c<nc; ++c ){
		for( v=0; v<fs; ++v ){
			for( u=0; u<fs; ++u ){
				*p_slider=c*pm+(v*ds+u)*st*enb; ++p_slider;
			}
		}
	}
	if(((fs*fs*nc)&7)!=0){
		for( i=nc*pm-(m<pm)*enb, pn-=n; pn>0; --pn ){ *p_slider=i; ++p_slider; }
	}
}

int conv_createOp( convOp_t* Op, unsigned int* p_temp, const cuda_context_t* p_ctx, unsigned int mask, int ds, int fs, int bat, int inc, int onc, int st )
{
	int prc, enb, pn, vs, i, k, s, tile_y, use_cmem, add_bias, atvop, os, anr, bnr, cnr, lda, ldb, ldc;		
	static const char* knames[][24]=
	{ 
		{
		    "d_sconv_128x32"     , "d_sconv_128x32_relu"     , "d_sconv_128x32_elu"     , "d_sconv_128x32_bias"     , "d_sconv_128x32_bias_relu"     , "d_sconv_128x32_bias_elu"     ,
		    "d_sconv_128x64"     , "d_sconv_128x64_relu"     , "d_sconv_128x64_elu"     , "d_sconv_128x64_bias"     , "d_sconv_128x64_bias_relu"     , "d_sconv_128x64_bias_elu"     ,
		    "d_sconv_128x128"    , "d_sconv_128x128_relu"    , "d_sconv_128x128_elu"    , "d_sconv_128x128_bias"    , "d_sconv_128x128_bias_relu"    , "d_sconv_128x128_bias_elu"    ,
		    "d_sconv_128x128_ldc", "d_sconv_128x128_relu_ldc", "d_sconv_128x128_elu_ldc", "d_sconv_128x128_bias_ldc", "d_sconv_128x128_bias_relu_ldc", "d_sconv_128x128_bias_elu_ldc"
		},
		{
		    "d_hconv_256x32"     , "d_hconv_256x32_relu"     , "d_hconv_256x32_elu"     , "d_hconv_256x32_bias"     , "d_hconv_256x32_bias_relu"     , "d_hconv_256x32_bias_elu"     ,
		    "d_hconv_256x64"     , "d_hconv_256x64_relu"     , "d_hconv_256x64_elu"     , "d_hconv_256x64_bias"     , "d_hconv_256x64_bias_relu"     , "d_hconv_256x64_bias_elu"     ,
		    "d_hconv_256x128"    , "d_hconv_256x128_relu"    , "d_hconv_256x128_elu"    , "d_hconv_256x128_bias"    , "d_hconv_256x128_bias_relu"    , "d_hconv_256x128_bias_elu"    ,
		    "d_hconv_256x128_ldc", "d_hconv_256x128_relu_ldc", "d_hconv_256x128_elu_ldc", "d_hconv_256x128_bias_ldc", "d_hconv_256x128_bias_relu_ldc", "d_hconv_256x128_bias_elu_ldc"
		}
	};
	cuda_kernel_t* p_kernel=&Op->kernel;
	prc=(mask>>1)&0x3;
	enb=prc?2:4;
	add_bias=(mask>>3)&0x1;
	atvop=mask>>4;
	os=(ds-fs)/st+1;
	anr=bat*ds*ds;
	bnr=inc*fs*fs;
	cnr=bat*os*os;
	lda=AFFIS(anr*enb,BASE_PITCH);
	ldb=AFFIS(bnr*enb,BASE_PITCH);
	ldc=AFFIS(cnr*enb,BASE_PITCH);
	Op->d_slider=0;
	Op->d_slider_cmem=0;
	if((onc<=32)|((onc>64)&(onc<=96))){
		i=0;
	} else {
		i=1+(((onc&127)==0)|((onc&127)>64));
	}
	use_cmem=((bnr<<2)<=p_ctx->cmemnb)&(i==2);
	pn=AFFIS(bnr,8);
	Op->slider_size=pn*sizeof(int);			
	if(cuMemAlloc( &Op->d_slider, Op->slider_size )!=CUDA_SUCCESS)
		return ERROR_OUT_OF_DEVICE_MEMORY;
	if(use_cmem){
		cuModuleGetGlobal( &Op->d_slider_cmem, NULL, p_ctx->module, "c_slider" );
	}
	generate_slider_conv( p_temp, anr*enb, lda, bnr, pn, ds, fs, inc, st, enb );
	cuMemcpyHtoD( Op->d_slider, p_temp, Op->slider_size );
	s=5+i;
	tile_y=1<<s;
	vs=AFFIS(cnr,(prc?4:2));
	i+=use_cmem;
	k=6*i+3*add_bias+atvop;
	i=k%6;
	cuda_context_create_kernel( p_kernel, p_ctx, knames[prc][k] );
	cuda_kernel_sao( p_kernel, k<18?(i<3?AM_4P_AS:AM_5P_AS):(i<3?AM_3P_AS:AM_4P_AS) );
	cuda_kernel_sbl( p_kernel, k<12?128:256, 1 );
	cuda_kernel_sgl( p_kernel, (vs+(prc?255:127))>>(prc?8:7), (onc+tile_y-1)>>s, 1 );
	if(use_cmem==0){
		cuda_kernel_sep_ptr( p_kernel, 3, Op->d_slider );
	}
	i=(use_cmem==0)+add_bias;
	cuda_kernel_sep_i32( p_kernel, 4+i, ldc );
	cuda_kernel_sep_i32( p_kernel, 5+i, ldb );
	cuda_kernel_sep_i32( p_kernel, 6+i, os  );
	cuda_kernel_sep_i32( p_kernel, 7+i, os  );
	cuda_kernel_sep_i32( p_kernel, 8+i, ds  );
	cuda_kernel_sep_i32( p_kernel, 9+i, ds  );
	cuda_kernel_sep_i32( p_kernel,10+i, pn  );
	cuda_kernel_sep_i32( p_kernel,11+i, vs  );
	cuda_kernel_sep_i32( p_kernel,12+i, onc );
	return SUCCESS;
}
void conv( convOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, const float* alpha, CUstream s )
{
	cuda_kernel_t* p=&Op->kernel;
	int b0=(Op->d_slider_cmem==0)&(Op->d_slider!=0);
	int b1=d_bias!=0;
	if(Op->d_slider_cmem!=0){
		cuMemcpyDtoDAsync( Op->d_slider_cmem, Op->d_slider, Op->slider_size, s );
	}
	cuda_kernel_sep_ptr( p, 0, d_c );
	cuda_kernel_sep_ptr( p, 1, d_a );
	cuda_kernel_sep_ptr( p, 2, d_b );
	if(b1){
		cuda_kernel_sep_ptr( p, 3+b0, d_bias );
	}
	cuda_kernel_sep_f32( p, 3+b0+b1, alpha!=NULL?*alpha:1.f );
	cuda_kernel_launch( p, s );
}
void conv_releaseOp( convOp_t* Op )
{
	if(Op->d_slider!=0){
	    cuMemFree(Op->d_slider);
	    Op->d_slider=0;
		Op->d_slider_cmem=0;
	}
}