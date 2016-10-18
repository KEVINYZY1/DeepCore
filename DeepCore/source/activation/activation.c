
#include"../../include/activation/activation.h"

void activation_createOp( activationOp_t* Op, cuda_context_t* p_ctx, int prc, int opcode, int n, int bt, int nc )
{
	static const char* knames[3][2][2]=
	{
		{
			{ "d_srelu_fprop", "d_selu_fprop" },
			{ "d_srelu_bprop", "d_selu_bprop" }
		},
		{
			{ "d_xrelu_fprop", "d_xelu_fprop" },
			{ "d_xrelu_bprop", "d_xelu_bprop" }
		},
		{
			{ "d_hrelu_fprop", "d_helu_fprop" },
			{ "d_hrelu_bprop", "d_helu_bprop" }
		}
	};
	static const unsigned int argmask_fprop[]={ AM_3P_3S, AM_3P_3S };
	static const unsigned int argmask_bprop[]={ AM_3P_3S, AM_4P_3S };

	const unsigned int size=bt*n*n;
	const unsigned int pitch=AFFIS(size,prc?(BASE_PITCH/2):(BASE_PITCH/4));
	const unsigned int bdx=1<<(5+(size>=64)+(size>=128)+(size>=256));
	const unsigned int gdx=(size+bdx-1)/bdx;
	Op->radix_ifunc.x=opcode;
	Op->radix_ifunc.y=prc?8:4;

	cuda_context_create_kernel( &Op->kernel[0], p_ctx, knames[prc][0][opcode] );
	cuda_kernel_sao( &Op->kernel[0], argmask_fprop[opcode] );
	cuda_kernel_sbl( &Op->kernel[0], bdx, 1  );
	cuda_kernel_sgl( &Op->kernel[0], gdx, nc );
	cuda_kernel_sep_i32( &Op->kernel[0], 3, pitch );
	cuda_kernel_sep_i32( &Op->kernel[0], 4, size  );

	cuda_context_create_kernel( &Op->kernel[1], p_ctx, knames[prc][1][opcode] );
	cuda_kernel_sao( &Op->kernel[1], argmask_bprop[opcode] );
	cuda_kernel_sbl( &Op->kernel[1], bdx, 1  );
	cuda_kernel_sgl( &Op->kernel[1], gdx, nc );
	cuda_kernel_sep_i32( &Op->kernel[1], 3+(opcode==1), pitch );
	cuda_kernel_sep_i32( &Op->kernel[1], 4+(opcode==1), size  );
}
void activation_fprop( activationOp_t* Op, CUdeviceptr d_dst, CUdeviceptr d_src, CUdeviceptr d_bias, float alpha, CUstream s )
{
	cuda_kernel_t* p=&Op->kernel[0];
	cuda_kernel_sep_ptr( p, 0, d_dst  );
	cuda_kernel_sep_ptr( p, 1, d_src  );
	cuda_kernel_sep_ptr( p, 2, d_bias );
	cuda_kernel_sep_f32( p, 5, alpha  );
	cuda_kernel_launch( p, s );
}
void activation_bprop( activationOp_t* Op, CUdeviceptr d_ydiff, CUdeviceptr d_ydata, CUdeviceptr d_xdiff, CUdeviceptr d_xdata, float alpha, CUstream s )
{
	cuda_kernel_t* p=&Op->kernel[1];
	int is_elu=Op->radix_ifunc.x==1;
	cuda_kernel_sep_ptr( p, 0, d_ydiff );
	if(is_elu){
		cuda_kernel_sep_ptr( p, 1, d_ydata );
	}
	cuda_kernel_sep_ptr( p, 1+is_elu, d_xdiff );
	cuda_kernel_sep_ptr( p, 2+is_elu, d_xdata );
	cuda_kernel_sep_f32( p, 5+is_elu, alpha   );
	cuda_kernel_launch( p, s );
}
