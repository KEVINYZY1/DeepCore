
#include"../../include/pooling/pooling.h"

int pooling_createOp( poolingOp_t* Op, const cuda_context_t* p_ctx, int is_max, int prc, int fn, int bt, int nc, int ps, int st )
{
	int bn, fsize, bsize, fpitch, bpitch, overlap, argmask, i, k, n, bdx, gdx;	
	const char* symbol[2];
	cuda_kernel_t* p;

	static const char* knames2x2[3][4]=
	{
		{ "d_sfpooling2x2_avg", "d_sbpooling2x2_avg", "d_sfpooling2x2_max", "d_sbpooling2x2_max" },
		{ "d_xfpooling2x2_avg", "d_hbpooling2x2_avg", "d_hfpooling2x2_max", "d_hbpooling2x2_max" },
		{ "d_hfpooling2x2_avg", "d_hbpooling2x2_avg", "d_hfpooling2x2_max", "d_hbpooling2x2_max" },
	};
	static const char* knames[3][2][4]=
	{
		{	
			{ "d_sfpooling_avg", "d_sbpooling_avg", "d_sfpooling_avg_ov", "d_sbpooling_avg_ov" },
			{ "d_sfpooling_max", "d_sbpooling_max", "d_sfpooling_max_ov", "d_sbpooling_max_ov" }
		},
		{
			{ "d_xfpooling_avg", "d_hbpooling_avg", "d_xfpooling_avg_ov", "d_xbpooling_avg_ov" },
			{ "d_hfpooling_max", "d_hbpooling_max", "d_hfpooling_max_ov", "d_xbpooling_max_ov" }
		},
		{	
			{ "d_hfpooling_avg", "d_hbpooling_avg", "d_hfpooling_avg_ov", "d_hbpooling_avg_ov" },
			{ "d_hfpooling_max", "d_hbpooling_max", "d_hfpooling_max_ov", "d_hbpooling_max_ov" }
		}
	};

	bn=(fn+st-1)/st;
	fsize=bt*fn*fn;
	bsize=bt*bn*bn;
	fpitch=AFFIS(fsize,64);
	bpitch=AFFIS(bsize,64);
	Op->d_max_id=0;
	if(is_max){
		int enb=((ps==2)&((fn&1)==0))?1:4;
		if(cuMemAlloc(&Op->d_max_id,bpitch*nc*enb)!=CUDA_SUCCESS)
			return ERROR_OUT_OF_DEVICE_MEMORY;
	}

	if((ps==2)&((fn&1)==0)){
		symbol[0]=knames2x2[prc][(is_max<<1)+0];
		symbol[1]=knames2x2[prc][(is_max<<1)+1];
		argmask=is_max?AM_3P_4S:AM_2P_4S;
	} else {
		overlap=ps>st;
		symbol[0]=knames[prc][is_max][(overlap<<1)+0];
		symbol[1]=knames[prc][is_max][(overlap<<1)+1];		
		argmask=is_max?(overlap?AM_3P_BS:AM_3P_9S):(overlap?AM_2P_BS:AM_2P_9S);
	}

	for( i=2+is_max, k=0; k<2; ++k )
	{
		p=&Op->kpooling[k];
		cuda_context_create_kernel( p, p_ctx, symbol[k] );
		cuda_kernel_sao( p, argmask );
		if(is_max){ cuda_kernel_sep_ptr( p, 2, Op->d_max_id ); }			
		cuda_kernel_sep_i32( p, i, fn	);
		if((ps==2)&((fn&1)==0))
		{	
			int s=k?(is_max?0:1):1;
			n=k?(is_max?fsize:bsize):bsize;
			cuda_kernel_sep_i32( p, i+1, fpitch>>s );
			cuda_kernel_sep_i32( p, i+2, bpitch    );
			cuda_kernel_sep_i32( p, i+3, n	       );
		}
		else
		{
			n=k?fsize:bsize;
			cuda_kernel_sep_i32( p, i+1, fn	    );
			cuda_kernel_sep_i32( p, i+2, fpitch );
			cuda_kernel_sep_i32( p, i+3, bn	    );
			cuda_kernel_sep_i32( p, i+4, bn	    );
			cuda_kernel_sep_i32( p, i+5, bpitch );
			cuda_kernel_sep_i32( p, i+6, bt	    );
			cuda_kernel_sep_i32( p, i+7, ps	    );
			cuda_kernel_sep_i32( p, i+8, ps	    );
			if(overlap){
				cuda_kernel_sep_i32( p, i+ 9, st );
				cuda_kernel_sep_i32( p, i+10, st );
			}
		}
		bdx=1<<(5+(n>32)+(n>64)+(n>128));
		gdx=(n+bdx-1)/bdx;		
		cuda_kernel_sgl( p, gdx, nc );
		cuda_kernel_sbl( p, bdx, 1  );
	}
	return SUCCESS;
}

void pooling_launch( poolingOp_t* Op, CUdeviceptr d_dst, CUdeviceptr d_src, int dir, CUstream s )
{	
	cuda_kernel_t* p=&Op->kpooling[dir];
	cuda_kernel_sep_ptr( p, 0, d_dst );
	cuda_kernel_sep_ptr( p, 1, d_src );
	cuda_kernel_launch( p, s );
}
void pooling_releaseOp( poolingOp_t* Op )
{
	if(Op->d_max_id){
		cuMemFree( Op->d_max_id );
		Op->d_max_id=0;
	}
}
