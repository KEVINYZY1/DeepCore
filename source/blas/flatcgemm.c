#include"../../include/blas/cgemm.h"

static void sflatcgemm_bk_b01( cuda_kernel_t* p_kernel, CUmodule module, int prc, int m, int bat, int onc )
{
    static const char* kname[][3]={{"dk_sflatcgemm_1x032","dk_sflatcgemm_1x064","dk_sflatcgemm_1x128"},{"dk_xflatcgemm_1x032","dk_xflatcgemm_1x064","dk_xflatcgemm_1x128"}};
    int i=(((onc>32)&(onc<=64))|((onc>128)&(onc<=192)))+((((onc>96)&(onc<=128))|(onc>192))<<1);
    int n=1<<(5+i);
    cuda_context_create_kernel( p_kernel, module, kname[prc][i] );
    cuda_kernel_sgl( p_kernel, m*((onc+n-1)/n), 1, 1 );
    cuda_kernel_sbl( p_kernel, n, 1 );
}
static void sflatcgemm_bk_b02( cuda_kernel_t* p_kernel, CUmodule module, int prc, int m, int bat, int onc )
{
    int i, n;
    static const char* kname[][3]={{"dk_sflatcgemm_2x032","dk_sflatcgemm_2x128","dk_sflatcgemm_2x256"},{"dk_xflatcgemm_2x032","dk_xflatcgemm_2x128","dk_xflatcgemm_2x256"}};
    if(onc<=96){ i=0; } else{ i=2-(((i>96)&(i<=128))|((i>256)&(i<=384))); }
    n=i==0?32:(i==1?128:256);
    cuda_context_create_kernel( p_kernel, module, kname[prc][i] );
    cuda_kernel_sgl( p_kernel, m*((onc+n-1)/n), 1, 1 );
    cuda_kernel_sbl( p_kernel, n, 1 );
}
static void sflatcgemm_bk_b04( cuda_kernel_t* p_kernel, CUmodule module, int prc, int m, int bat, int onc )
{
    cuda_context_create_kernel( p_kernel, module, prc==0?"dk_sflatcgemm_4x32":"dk_xflatcgemm_4x32" );
    cuda_kernel_sgl( p_kernel, m*((onc+31)>>5), 1, 1 );
    cuda_kernel_sbl( p_kernel, 64, 1 );
}
static void sflatcgemm_bk_b08( cuda_kernel_t* p_kernel, CUmodule module, int prc, int m, int bat, int onc )
{
    static const char* kname[][2]={{"dk_sflatcgemm_8x32","dk_sflatcgemm_8x64"},{"dk_xflatcgemm_8x32","dk_xflatcgemm_8x64"}};
    int i=((onc>32)&(onc<=64))|(onc>96);
    int n=1<<(5+i);
    cuda_context_create_kernel( p_kernel, module, kname[prc][i] );
    cuda_kernel_sgl( p_kernel, ((bat+7)>>3)*((onc+n-1)/n), m, 1 );
    cuda_kernel_sbl( p_kernel, 128, 1 );
}
static void sflatcgemm_bk_b16( cuda_kernel_t* p_kernel, CUmodule module, int prc, int m, int bat, int onc )
{
    cuda_context_create_kernel( p_kernel, module, prc==0?"dk_sflatcgemm_16x32":"dk_xflatcgemm_16x32" );
    cuda_kernel_sgl( p_kernel, ((bat+15)>>4)*((onc+31)>>5), m, 1 );
    cuda_kernel_sbl( p_kernel, 128, 1 );
}
static void sflatcgemm_bk_b32( cuda_kernel_t* p_kernel, CUmodule module, int prc, int m, int bat, int onc )
{
    cuda_context_create_kernel( p_kernel, module, prc==0?"dk_sflatcgemm_32x32":"dk_xflatcgemm_32x32" );
    cuda_kernel_sgl( p_kernel, ((bat+31)>>5)*((onc+31)>>5), m, 1 );
    cuda_kernel_sbl( p_kernel, 256, 1 );
}
static void (*p_sflatcgemm_bk[])( cuda_kernel_t*, CUmodule module, int, int, int, int )=
{
    sflatcgemm_bk_b01,
    sflatcgemm_bk_b02,
    sflatcgemm_bk_b04,
    sflatcgemm_bk_b08,
    sflatcgemm_bk_b16,
    sflatcgemm_bk_b32
};
static void sflatcgemm_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, int slice_size, int bat, int pnc, int qnc, int dir, int prc )
{
    int i, inc, onc, n;
    if((bat>16)&(bat<=24)){ i=3; } else
    if((bat>32)&(bat<=48)){ i=4; } else {
        i=idc_bhs(idc_minls(bat));
        i=i<=5?i:5;
    }
    inc=dir?qnc:pnc;
    onc=dir?pnc:qnc;
    n=pnc*slice_size;
    p_sflatcgemm_bk[i]( p_kernel, module, prc, slice_size>>4, bat, onc );
    cuda_kernel_sao( p_kernel, AM_3P_7S );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f              );
    cuda_kernel_sep_i32( p_kernel, 4, slice_size       );
    cuda_kernel_sep_i32( p_kernel, 5, bat              );
    cuda_kernel_sep_i32( p_kernel, 6, inc              );
    cuda_kernel_sep_i32( p_kernel, 7, onc              );
    cuda_kernel_sep_i32( p_kernel, 8, dir?slice_size:n );
    cuda_kernel_sep_i32( p_kernel, 9, dir?n:slice_size );
}

static void hflatcgemm_bk_b01( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
    static const char* kname[][2]={{"dk_hflatcgemmnn_1x32","dk_hflatcgemmnn_1x64"},{"dk_hflatcgemmnt_1x32","dk_hflatcgemmnt_1x64"}};
    int i=(onc>32)&(onc!=96);
    cuda_context_create_kernel( p_kernel, module, kname[dir][i] );
    cuda_kernel_sgl( p_kernel, m*((onc+31)>>5), 1, 1 );
    cuda_kernel_sbl( p_kernel, 32, 1 );
}
static void hflatcgemm_bk_b02( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
	static const char* kname[][2]={{"dk_hflatcgemmnn_2x32","dk_hflatcgemmnn_2x64"},{"dk_hflatcgemmnt_2x32","dk_hflatcgemmnt_2x64"}};
	int i=(onc>32)&(onc!=96);
	cuda_context_create_kernel( p_kernel, module, kname[dir][i] );
	cuda_kernel_sgl( p_kernel, m*((onc+31)>>5), 1, 1 );
	cuda_kernel_sbl( p_kernel, 32, 1 );
}
static void hflatcgemm_bk_b04( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
    cuda_context_create_kernel( p_kernel, module, dir==0?"dk_hflatcgemmnn_4x32":"dk_hflatcgemmnt_4x32" );
    cuda_kernel_sgl( p_kernel, m*((onc+31)>>5), 1, 1 );
    cuda_kernel_sbl( p_kernel, 64, 1 );
}
static void hflatcgemm_bk_b08( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
    static const char* kname[][2]={{"dk_hflatcgemmnn_8x32","dk_hflatcgemmnn_8x64"},{"dk_hflatcgemmnt_8x32","dk_hflatcgemmnt_8x64"}};
    int i=((onc>32)&(onc<=64))|(onc>96);
    int n=1<<(5+i);
    cuda_context_create_kernel( p_kernel, module, kname[dir][i] );
    cuda_kernel_sgl( p_kernel, ((bat+7)>>3)*((onc+n-1)/n), m, 1 );
    cuda_kernel_sbl( p_kernel, 128, 1 );
}
static void hflatcgemm_bk_b16( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
    static const char* kname[][3]={{"dk_hflatcgemmnn_16x32","dk_hflatcgemmnn_16x64","dk_hflatcgemmnn_16x128"},{"dk_hflatcgemmnt_16x32","dk_hflatcgemmnt_16x64","dk_hflatcgemmnt_16x128"}};
    int tile_y, i=0;
    if(((onc>32)&(onc<=64))|((onc>128)&(onc<=192))){ i=1; } else
    if(((onc>96)&(onc<=128))|(onc>192)){ i=2; }
    tile_y=1<<(5+i);
    cuda_context_create_kernel( p_kernel, module, kname[dir][i] );
    cuda_kernel_sgl( p_kernel, ((bat+15)>>4)*((onc+tile_y-1)/tile_y), m, 1 );
    cuda_kernel_sbl( p_kernel, i==0?128:256, 1 );
}
static void hflatcgemm_bk_b32( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
    static const char* kname[][2]={{"dk_hflatcgemmnn_32x32","dk_hflatcgemmnn_32x64"},{"dk_hflatcgemmnt_32x32","dk_hflatcgemmnt_32x64"}};
    int i=(onc<=32)|((onc>64)&(onc<=96));
    int tile_y=1<<(6-i);
    cuda_context_create_kernel( p_kernel, module, kname[dir][i] );
    cuda_kernel_sgl( p_kernel, ((bat+31)>>5)*((onc+tile_y-1)/tile_y), m, 1 );
    cuda_kernel_sbl( p_kernel, 256, 1 );
}
static void hflatcgemm_bk_b64( cuda_kernel_t* p_kernel, CUmodule module, int m, int bat, int onc, int dir )
{
    cuda_context_create_kernel( p_kernel, module, dir==0?"dk_hflatcgemmnn_64x32":"dk_hflatcgemmnt_64x32" );
    cuda_kernel_sgl( p_kernel, ((bat+63)>>6)*((onc+31)>>5), m, 1 );
    cuda_kernel_sbl( p_kernel, 256, 1 );
}
static void (*p_hflatcgemm_bk[])( cuda_kernel_t*, CUmodule module, int, int, int, int )=
{
    hflatcgemm_bk_b01,
    hflatcgemm_bk_b02,
    hflatcgemm_bk_b04,
    hflatcgemm_bk_b08,
    hflatcgemm_bk_b16,
    hflatcgemm_bk_b32,
    hflatcgemm_bk_b64
};
static void hflatcgemm_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, int slice_size, int bat, int pnc, int qnc, int dir )
{
    int i;
    if((bat>16)&(bat<=24)){ i=3; } else
    if((bat>32)&(bat<=48)){ i=4; } else {
        i=idc_bhs(idc_minls(bat));
        i=i<=5?i:5;
    }
    p_hflatcgemm_bk[i]( p_kernel, module, slice_size>>4, bat, dir?pnc:qnc, dir );
    cuda_kernel_sao( p_kernel, AM_3P_5S );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f        );
    cuda_kernel_sep_i32( p_kernel, 4, slice_size );
    cuda_kernel_sep_i32( p_kernel, 5, bat        );
    cuda_kernel_sep_i32( p_kernel, 6, pnc        );
    cuda_kernel_sep_i32( p_kernel, 7, qnc        );
}

void idc_flatcgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int prc, int slice_size, int bat, int pnc, int qnc, int dir )
{
    CUmodule module=p_ctx->module_fftconv;
    if(prc!=1){
        sflatcgemm_create_kernel( p_kernel, module, slice_size, bat, pnc, qnc, dir, prc!=0 );
    } else {
        hflatcgemm_create_kernel( p_kernel, module, slice_size, bat, pnc, qnc, dir );
    }
}

