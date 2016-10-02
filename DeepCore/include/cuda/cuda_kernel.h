#ifndef __cuda_kernel_h__
#define __cuda_kernel_h__

#include<cuda.h>
#include<vector_types.h>
#include"../dc_macro.h"
#include"../dc_argmask.h"

typedef struct cuda_kernel{
	CUfunction	id;	
	unsigned int	gdx;
	unsigned int	gdy;
	ushort2		block;
	unsigned int	smemnb;
	unsigned int	arg_size;
	void*		extra[5];
	unsigned char	arg_ofs[16];
	char		args[128];
} cuda_kernel_t;

__forceinline void cuda_kernel_sao( cuda_kernel_t* p, unsigned int mask )
{	
	unsigned int i=0, ofs=0, k=mask;
	do{
		p->arg_ofs[i++]=(unsigned char)ofs;
		if((k&0x3)==PA){
			ofs=AFFIS(ofs,__alignof(CUdeviceptr)); ofs+=sizeof(CUdeviceptr); 
		} else {
			ofs=AFFIS(ofs,__alignof(int)); ofs+=__alignof(int);
		}
	}while((k>>=2)!=0);
	p->arg_size=ofs;
}
__forceinline void cuda_kernel_set_smemnb( cuda_kernel_t* p_kernel, unsigned int nb )
{
	p_kernel->smemnb=nb;
}
__forceinline void cuda_kernel_sgl( cuda_kernel_t* p_kernel, unsigned int gdx, unsigned int gdy )
{
	p_kernel->gdx=gdx; p_kernel->gdy=gdy;
}
__forceinline void cuda_kernel_sbl( cuda_kernel_t* p_kernel, unsigned int bdx, unsigned int bdy )
{
	p_kernel->block.x=bdx; p_kernel->block.y=bdy;
}
__forceinline void cuda_kernel_sep_ptr( cuda_kernel_t* p_kernel, int i, CUdeviceptr p )
{
	*((CUdeviceptr*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
__forceinline void cuda_kernel_sep_i32( cuda_kernel_t* p_kernel, int i, int p )
{
	*((int*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
__forceinline void cuda_kernel_sep_f32( cuda_kernel_t* p_kernel, int i, float p )
{
	*((float*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
__forceinline CUresult cuda_kernel_launch( cuda_kernel_t* p, CUstream s )
{
	return cuLaunchKernel( p->id, p->gdx, p->gdy, 1, p->block.x, p->block.y, 1, p->smemnb, s, NULL, (void**)p->extra );
}

#endif
