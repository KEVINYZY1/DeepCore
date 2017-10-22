#include"../../include/bnorm/bnorm.h"

void idc_bnorm_createOp( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, uint32_t mask, int size, int pitch, int nc )
{
	static const char* s_radix[]={ "_032", "_064", "_128", "_256" };
	char kname[32]={ "dk_" };
	int dir=mask&0x1;
	int prc=(mask>>1)&0x3;
	int o=dir==0?6:8;
	int b=prc==2;
	int psize=(size+b)>>(prc==2);
	float rsamples=1.f/size;
	int i=4;
	if((size*4)<=(p_ctx->max_smemnb_per_block-(64*(psize>2048)))){
		--i;
		i-=(psize<=8192);
		i-=(psize<=4096);
		i-=(psize<=2048);
	}
	idc_strcat( kname, prc==0?"s":(prc==1?"h":"x") );
	idc_strcat( kname, "bn" );
	if(dir!=0){ idc_strcat( kname, "_grad" ); }
	if(i<4){ idc_strcat( kname, s_radix[i] ); }
	cuda_context_create_kernel( p_kernel, p_ctx->module, kname );
	if(i<4){ cuda_kernel_set_smemnb( p_kernel, size*4 ); }
	cuda_kernel_sbl( p_kernel, 1<<(5+i-(i==4)), 1 );
	cuda_kernel_sgl( p_kernel, nc, 1, 1 );
	cuda_kernel_sao( p_kernel, dir==0?AM_6P_3S:AM_8P_3S );
	cuda_kernel_sep_i32( p_kernel, o++, psize    );
	cuda_kernel_sep_i32( p_kernel, o++, pitch    );
	cuda_kernel_sep_f32( p_kernel, o++, rsamples );
}
void idc_bnorm( cuda_kernel_t* p_kernel, CUdeviceptr d_dy, CUdeviceptr d_mean, CUdeviceptr d_vari, CUdeviceptr d_x, CUdeviceptr d_gamm, CUdeviceptr d_beta, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_dy   );
	cuda_kernel_sep_ptr( p_kernel, 1, d_mean );
	cuda_kernel_sep_ptr( p_kernel, 2, d_vari );
	cuda_kernel_sep_ptr( p_kernel, 3, d_x    );
	cuda_kernel_sep_ptr( p_kernel, 4, d_gamm );
	cuda_kernel_sep_ptr( p_kernel, 5, d_beta );
	cuda_kernel_launch( p_kernel, s );
}
void idc_bnorm_grad( cuda_kernel_t* p_kernel, CUdeviceptr d_dx, CUdeviceptr d_dg, CUdeviceptr d_db, CUdeviceptr d_x, CUdeviceptr d_dy, CUdeviceptr d_mean, CUdeviceptr d_vari, CUdeviceptr d_gamm, CUstream s )
{
	cuda_kernel_sep_ptr( p_kernel, 0, d_dx   );
	cuda_kernel_sep_ptr( p_kernel, 1, d_dg   );
	cuda_kernel_sep_ptr( p_kernel, 2, d_db   );
	cuda_kernel_sep_ptr( p_kernel, 3, d_x    );
	cuda_kernel_sep_ptr( p_kernel, 4, d_dy   );
	cuda_kernel_sep_ptr( p_kernel, 5, d_gamm );
	cuda_kernel_sep_ptr( p_kernel, 6, d_mean );
	cuda_kernel_sep_ptr( p_kernel, 7, d_vari );
	cuda_kernel_launch( p_kernel, s );
}