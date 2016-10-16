#include"../../include/conv/conv.h"
#include"../../include/blas/gemm.h"

static void __generate_slider( unsigned int* p_slider, int m, int pm, int n, int pn, int ds, int fs, int inc, int st, int enb )
{
	int u, v, c, i;
	for( c=0; c<inc; ++c )
	{
		for( v=0; v<fs; ++v ){
			for( u=0; u<fs; ++u ){
				*p_slider=c*pm+(v*st*ds+u*st)*enb; ++p_slider;
			}
		}
	}
	if(((fs*fs*inc)&7)!=0){
		for( i=inc*pm-(m<pm)*enb, pn-=n; pn>0; --pn ){ *p_slider=i; ++p_slider; }
	}
}

int conv_createOp( convOp_t* Op, unsigned int* p_temp, const cuda_context_t* p_ctx, unsigned int mask, int ds, int fs, int bat, int inc, int onc, int st )
{
	int prc, enb, add_bias, atvop, os, anr, bnr, cnr, lda, ldb, ldc;
	cuda_kernel_t* p_kernel=&Op->kernel;

	prc=mask&0x3;
	enb=prc?2:4;
	add_bias=(mask>>2)&0x1;
	atvop=(mask>>3)+1;
	os=(ds-fs)/st+1;
	anr=bat*ds*ds;
	bnr=inc*fs*fs;
	cnr=bat*os*os;
	lda=AFFIS(anr*enb,BASE_PITCH);
	ldb=AFFIS(bnr*enb,BASE_PITCH);
	ldc=AFFIS(cnr*enb,BASE_PITCH);
	Op->d_slider=0;
	Op->flag=0;
	if((ds!=fs)&(fs>1))
	{
		int pn, vs, i, k, s, tile_y;
		static const char* knames[]=
		{ 
			"d_sconv_128x32"        , "d_sconv_128x32_relu"        , "d_sconv_128x32_elu"        , "d_sconv_128x32_bias"        , "d_sconv_128x32_bias_relu"        , "d_sconv_128x32_bias_elu"        ,
			"d_sconv_128x32_bc"     , "d_sconv_128x32_relu_bc"     , "d_sconv_128x32_elu_bc"     , "d_sconv_128x32_bias_bc"     , "d_sconv_128x32_bias_relu_bc"     , "d_sconv_128x32_bias_elu_bc"     ,
			"d_sconv_128x64"        , "d_sconv_128x64_relu"        , "d_sconv_128x64_elu"        , "d_sconv_128x64_bias"        , "d_sconv_128x64_bias_relu"        , "d_sconv_128x64_bias_elu"        ,
			"d_sconv_128x64_bc"     , "d_sconv_128x64_relu_bc"     , "d_sconv_128x64_elu_bc"     , "d_sconv_128x64_bias_bc"     , "d_sconv_128x64_bias_relu_bc"     , "d_sconv_128x64_bias_elu_bc"     ,
			"d_sconv_128x128"       , "d_sconv_128x128_relu"       , "d_sconv_128x128_elu"       , "d_sconv_128x128_bias"       , "d_sconv_128x128_bias_relu"       , "d_sconv_128x128_bias_elu"       ,
			"d_sconv_128x128_bc"    , "d_sconv_128x128_relu_bc"    , "d_sconv_128x128_elu_bc"    , "d_sconv_128x128_bias_bc"    , "d_sconv_128x128_bias_relu_bc"    , "d_sconv_128x128_bias_elu_bc"    ,
			"d_sconv_128x128_ldc"   , "d_sconv_128x128_relu_ldc"   , "d_sconv_128x128_elu_ldc"   , "d_sconv_128x128_bias_ldc"   , "d_sconv_128x128_bias_relu_ldc"   , "d_sconv_128x128_bias_elu_ldc"   ,
			"d_sconv_128x128_ldc_bc", "d_sconv_128x128_relu_ldc_bc", "d_sconv_128x128_elu_ldc_bc", "d_sconv_128x128_bias_ldc_bc", "d_sconv_128x128_bias_relu_ldc_bc", "d_sconv_128x128_bias_elu_ldc_bc"
		};
		i=(onc>32)+(((onc&127)==0)|((onc&127)>64));
		Op->flag=(p_ctx->arch>=50)&((inc*fs*fs)<=(p_ctx->cmemnb-128))&(i==2);
		pn=AFFIS(bnr,8);
		if(Op->flag){
			cuModuleGetGlobal( &Op->d_slider, NULL, p_ctx->module, "c_slider" );
		} else {
			if(cuMemAlloc( &Op->d_slider, sizeof(int)*pn )!=CUDA_SUCCESS)
				return ERROR_OUT_OF_DEVICE_MEMORY;
		}
		__generate_slider( p_temp, anr*enb, lda, bnr, pn, ds, fs, inc, st, enb );
		cuMemcpyHtoD( Op->d_slider, p_temp, sizeof(int)*pn );
		s=5+i;
		tile_y=1<<s;
		vs=AFFIS(cnr,2);
		i+=Op->flag;
		k=12*i+6*((onc&(tile_y-1))!=0)+3*add_bias+atvop;
		i=k%6;
		cuda_context_create_kernel( p_kernel, p_ctx, knames[k] );
		cuda_kernel_sao( p_kernel, k<36?(i<3?AM_4P_BS:AM_5P_BS):(i<3?AM_3P_BS:AM_4P_BS) );
		cuda_kernel_sbl( p_kernel, k<24?128:256, 1 );
		cuda_kernel_sgl( p_kernel, (vs+127)>>7, (onc+tile_y-1)>>s );
		i=(Op->flag==0)+add_bias;
		cuda_kernel_sep_i32( p_kernel, 5+i, ldc );
		cuda_kernel_sep_i32( p_kernel, 6+i, ldb );
		cuda_kernel_sep_i32( p_kernel, 7+i, os  );
		cuda_kernel_sep_i32( p_kernel, 8+i, os  );
		cuda_kernel_sep_i32( p_kernel, 9+i, ds  );
		cuda_kernel_sep_i32( p_kernel,10+i, ds  );
		cuda_kernel_sep_i32( p_kernel,11+i, pn  );
		cuda_kernel_sep_i32( p_kernel,12+i, vs  );
		cuda_kernel_sep_i32( p_kernel,13+i, onc );
	} else
	if(fs==1){
		gemm_create_kernel( p_kernel, p_ctx, prc, anr, bnr, onc, lda, ldb, ldc );
	}
	return SUCCESS;
}
void conv( convOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, float alpha, float beta, CUstream s )
{
	cuda_kernel_t* p=&Op->kernel;
	int b0=(Op->flag==0)&(Op->d_slider!=0);
	int b1=d_bias!=0;
	cuda_kernel_sep_ptr( p, 0, d_c );
	cuda_kernel_sep_ptr( p, 1, d_a );
	cuda_kernel_sep_ptr( p, 2, d_b );
	if(b1){
		cuda_kernel_sep_ptr( p, 3, d_bias );
	}
	if(b0){
	    cuda_kernel_sep_ptr( p, 3+b1, Op->d_slider );
	}
	cuda_kernel_sep_f32( p, 3+b0+b1, alpha );
	cuda_kernel_sep_f32( p, 4+b0+b1, beta  );
	cuda_kernel_launch( p, s );
}
void conv_releaseOp( convOp_t* Op )
{
	if((Op->flag==0)&(Op->d_slider!=0)){
	    cuMemFree(Op->d_slider);
	    Op->d_slider=0;
	}
}