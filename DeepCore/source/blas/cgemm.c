#include"../../include/blas/cgemm.h"
#include"../../include/fft/fft_bop.h"

void cgemm_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int bat, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
	static const char* knames[4][5]=
	{
		{"d_scgemmBatched_32x32", "d_scgemmBatched_32x32_abc", "d_scgemmBatched_32x32_cbc", "d_scgemmBatched_32x32_acbc", "d_scgemmBatched_32x32_abcbc"},
		{"d_scgemmBatched_32x64", "d_scgemmBatched_32x64_abc", "d_scgemmBatched_32x64_cbc", "d_scgemmBatched_32x64_acbc", "d_scgemmBatched_32x64_abcbc"},
		{"d_scgemmBatched_64x32", "d_scgemmBatched_64x32_abc", "d_scgemmBatched_64x32_cbc", "d_scgemmBatched_64x32_acbc", "d_scgemmBatched_64x32_abcbc"},
		{"d_scgemmBatched_64x64", "d_scgemmBatched_64x64_abc", "d_scgemmBatched_64x64_cbc", "d_scgemmBatched_64x64_acbc", "d_scgemmBatched_64x64_abcbc"}
	};
	static const unsigned char block_size[]={63,127,127,255};
	static const unsigned char dx[]={32,32,64,64};
	static const unsigned char dy[]={32,64,32,64};
	static const unsigned char shfl_a[][5]={{1,1,1,1,0},{2,1,2,2,0}};
	static const unsigned char shfl_b[][5]={{1,1,1,1,0},{2,2,1,2,0}};
	unsigned int y=((anr>32)<<1)+(cnc>32);
	unsigned int nx=dx[y];
	unsigned int ny=dy[y];
	unsigned int x=((bnr&7)==0)?(((anr&(ny-1))!=0)+(((cnc&(nx-1))!=0)<<1)):4;
	unsigned int nbx=(anr+nx-1)/nx;
	unsigned int nby=(cnc+ny-1)/ny;
	unsigned int gdx=(y==0)?bat:(nbx*nby);
	unsigned int gdy=(y==0)?1:bat;
	unsigned int i=4+(y==3);
	lda>>=shfl_a[prc][x];
	ldb>>=shfl_b[prc][x];
	cuda_context_create_kernel( p, p_ctx, knames[y][x] );
	cuda_kernel_sao( p, y<3?AM_3P_7S:AM_3P_8S );
	cuda_kernel_sbl( p, block_size[y]+1, 1 );
	cuda_kernel_sgl( p, gdx, gdy, 1 );
	cuda_kernel_sep_f32( p, 3, 1.f );
	if(y==3){
		cuda_kernel_sep_i32( p, 4, nbx );
	}
	cuda_kernel_sep_i32( p, i+0, anr );
	cuda_kernel_sep_i32( p, i+1, bnr );
	cuda_kernel_sep_i32( p, i+2, cnc );
	cuda_kernel_sep_i32( p, i+3, lda );
	cuda_kernel_sep_i32( p, i+4, ldb );
	cuda_kernel_sep_i32( p, i+5, ldc );
}
void cgemm_flat_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int slice_size, int bat, int inc, int onc, int dir )
{
	static const char* knames[]=
	{
		"d_scgemm_flat_01_0016",
		"d_scgemm_flat_01_0032",
		"d_scgemm_flat_01_0064",
		"d_scgemm_flat_01_0128",
		"d_scgemm_flat_01_0256",
		"d_scgemm_flat_01_0512",
		"d_scgemm_flat_01_1024",
		"d_scgemm_flat_02_0016",
		"d_scgemm_flat_02_0032",
		"d_scgemm_flat_02_0064",
		"d_scgemm_flat_02_0128",
		"d_scgemm_flat_02_0256",
		"d_scgemm_flat_02_0512",
		"d_scgemm_flat_04_0016",
		"d_scgemm_flat_04_0032",
		"d_scgemm_flat_04_0064",
		"d_scgemm_flat_04_0128",
		"d_scgemm_flat_04_0256",
		"d_scgemm_flat_08_0016",
		"d_scgemm_flat_08_0032",
		"d_scgemm_flat_08_0064",
		"d_scgemm_flat_08_0128",
		"d_scgemm_flat_16_0016",
		"d_scgemm_flat_16_0032",
		"d_scgemm_flat_16_0064",
		"d_scgemm_flat_lb_0032",
		"d_scgemm_flat_lb_0064",
		"d_scgemm_flat_lb_0128"
	};	
	static const unsigned char block_size[]={
		 63, 63,127,255,255,255,255,
		127,127,255,255,255,255,
		 63,127,255,255,255,
		127,127,255,255,
		255,255,255,
		255,255,255
	};
	static const short nmax[]={1024,512,256,128,64,128};
	static const unsigned char ofs[]={0,7,13,18,22,25};
	int axis=__bffs(bat>=32?32:bat);
	int i=ofs[axis];
	int m=nmax[axis];
	int r=__bffs(fft_get_exec_size(onc>m?m:onc))-4-(bat>16);
	int k=i+r;
	int n=slice_size*(dir?onc:inc);
	int n_tiles=slice_size>>4;
	int tile=1<<(4+r+(bat>16));
	int d=onc>tile?((onc+tile-1)/tile):1;
	cuda_context_create_kernel( p, p_ctx, knames[k] );
	cuda_kernel_sao( p, AM_3P_6S );
	cuda_kernel_sbl( p, block_size[k]+1, 1 );
	if(bat<32){
		cuda_kernel_sgl( p, n_tiles, d, 1 );
	} else {
		cuda_kernel_sgl( p, bat>>(5-r), n_tiles, d );
	}
	cuda_kernel_sep_f32( p, 3, 1.f              );
	cuda_kernel_sep_i32( p, 4, slice_size       );
	cuda_kernel_sep_i32( p, 5, inc              );
	cuda_kernel_sep_i32( p, 6, onc              );
	cuda_kernel_sep_i32( p, 7, dir?slice_size:n );
	cuda_kernel_sep_i32( p, 8, dir?n:slice_size );
}
void cgemv_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int bat, int nr, int nc, int lda, int ldb, int ldc )
{
	static const char* knames[][2]=
	{ 
		{"d_scgemv", "d_scgemv_bc"},
		{"d_xcgemv", "d_xcgemv_bc"},
		{"d_hcgemv", "d_hcgemv_bc"}
	};
	int i=((nr&127)|(nc&15))!=0;
	cuda_context_create_kernel( p, p_ctx, knames[prc][i] );
	cuda_kernel_sao( p, AM_3P_6S );
	cuda_kernel_sgl( p, (nr+127)>>7, bat, 1 );
	cuda_kernel_sbl( p, 128, 1 );
	cuda_kernel_sep_f32( p, 3, 1.f );
	cuda_kernel_sep_i32( p, 4, nr  );
	cuda_kernel_sep_i32( p, 5, nc  );
	cuda_kernel_sep_i32( p, 6, lda );
	cuda_kernel_sep_i32( p, 7, ldb );
	cuda_kernel_sep_i32( p, 8, ldc );
}
void cgevv_create_kernel( cuda_kernel_t* p, const cuda_context_t* p_ctx, int prc, int bat, int na, int nb, int lda, int ldb, int ldc )
{
	int enb, i, gdx, gdy, bdx, bdy;
	static const char* knames[][3]=
	{
		{ "d_scgevv_syncfree", "d_scgevv_block", "d_scgevv" }
	};
	enb=prc?4:8;
	if((na<=32)&(nb<=p_ctx->max_smemnb_per_block/(2*enb))){
		i=0; gdx=(bat+1)>>1; gdy=1; bdx=32; bdy=2;
	} else 
	if(((int)na<=p_ctx->max_block_size)&(nb<=p_ctx->max_smemnb_per_block/enb)){
		i=1; gdx=bat; gdy=1; bdx=fft_get_exec_size(na); bdy=1;
	} else {
		i=2; gdx=(na+255)>>8; gdy=bat; bdx=256; bdy=1;
	}
	cuda_context_create_kernel( p, p_ctx, knames[prc][i] );
	cuda_kernel_sao( p, AM_3P_6S );
	cuda_kernel_sgl( p, gdx, gdy, 1 );
	cuda_kernel_sbl( p, bdx, bdy );
	if(i<2){
		cuda_kernel_set_smemnb( p, nb*bdy*enb );
	}
	cuda_kernel_sep_f32( p, 3, 1.f  );
	cuda_kernel_sep_i32( p, 4, na	);
	cuda_kernel_sep_i32( p, 5, nb	);
	cuda_kernel_sep_i32( p, 6, lda	);
	cuda_kernel_sep_i32( p, 7, ldb	);
	cuda_kernel_sep_i32( p, 8, ldc	);
}
void cgemm( cuda_kernel_t* p, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float alpha, CUstream s )
{
	cuda_kernel_sep_ptr( p, 0, d_c );
	cuda_kernel_sep_ptr( p, 1, d_a );
	cuda_kernel_sep_ptr( p, 2, d_b );
	if(alpha!=1.f){
		*((float*)&p->args[p->arg_ofs[3]])*=alpha;
	}
	cuda_kernel_launch( p, s );
}