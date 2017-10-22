#include"../../include/idc_bitop.h"
#include"../../include/conv/fftconv.h"

size_t idc_fftconv_createOp( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat, int pad_x, int pad_y )
{
    int dir, prc, b, c, p, q, fft_size, is_opt, axis, snx, lds, sny, onx, ony, ldo, nx, ny, dx, dy, n, grid_x, grid_y, n_cells_per_chl, radix, is_split;
    cuda_kernel_t* p_kernel;
    
    dir=mask&0x1;
    snx=dir?qnx:pnx;
    sny=dir?qny:pny;
    lds=dir?ldq:ldp;
    onx=dir?pnx:qnx;
    ony=dir?pny:qny;
    ldo=dir?ldp:ldq;
    nx=snx+(pad_x<<1);
    ny=sny+(pad_y<<1);
    n=nx>ny?nx:ny;	
    if(n<=32)
    {
        fft_size=idc_minls(n);
        fft_size=fft_size<=8?8:fft_size;
        axis=idc_bhs(fft_size)-3;
        grid_x=grid_y=1;
        is_split=0;
    } 
    else 
    {
        fft_size=32;
        axis=2;
        dx=33-fnx;
        dy=33-fny;
        grid_x=(onx+dx-1)/dx;
        grid_y=(ony+dy-1)/dy;
        is_split=1;
        if(n<=128){
            int aa=(pnc*qnc+(pnc+qnc)*bat)*idc_minls(n);
            int ba=(pnc*qnc+(pnc+qnc)*bat*grid_x*grid_y)*32;
            is_split=(ba>aa)|(idc_popc(grid_x)>1)|(idc_popc(grid_y)>1);
            if(is_split==0){
                fft_size=idc_minls(n);
                axis=idc_bhs(fft_size)-3;
                grid_x=grid_y=1;
            }
        }
    }
    n_cells_per_chl=bat*grid_x*grid_y;
    if((fft_size<=32)&((n_cells_per_chl==1)|(n_cells_per_chl>8)))
        return idc_cellconv_createOp( Op, p_ctx, mask, pnx, pny, pnc, ldp, fnx, fny, qnx, qny, qnc, ldq, bat, pad_x, pad_y );
    prc=(mask>>1)&0x3;
    b=prc&1;
    p=(dir?qnc:pnc)>>b;
    q=(dir?pnc:qnc)>>b;	
    is_opt=((fnx==3)&(fny==3))|((fnx==5)&(fny==5))|((fnx==7)&(fny==7));
    c=axis<3;
    radix=fft_size>8?8:16;
    
    {
        int is_pad=(pad_x!=0)|(pad_y!=0);
        int is_ext=((snx!=fft_size)|(sny!=fft_size))&(1^is_pad);
        p_kernel=&Op->kfft[0];
        idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, is_split, is_ext, is_pad, 0 );
        n=p*n_cells_per_chl;
        cuda_kernel_sgl( p_kernel, c?((n+radix-1)/radix):bat, c?1:p, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, snx );
        cuda_kernel_sep_i32( p_kernel, 4, sny );
        cuda_kernel_sep_i32( p_kernel, 5, lds );
        if(c){
            cuda_kernel_sep_i32( p_kernel, 6, n_cells_per_chl );
            if(is_split){			
                cuda_kernel_sep_i32( p_kernel, 7, n      );
                cuda_kernel_sep_i32( p_kernel, 8, grid_x );
                cuda_kernel_sep_i32( p_kernel, 9, grid_y );
                cuda_kernel_sep_i32( p_kernel,10, dx     );
                cuda_kernel_sep_i32( p_kernel,11, dy     );
            }
        }			
        if(is_pad){
            cuda_kernel_sep_i32( p_kernel, is_split?12:(axis<3?7:6), pad_x );
            cuda_kernel_sep_i32( p_kernel, is_split?13:(axis<3?8:7), pad_y );
        }
    }
    
    {
        p_kernel=&Op->kfft[1];
        if((fft_size>8)&(fft_size<=64)&is_opt){
            idc_create_fft_kernel_r2c_opt( p_kernel, p_ctx, axis, prc, dir, (fnx>3)+(fnx>5) );
        } else {
            idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, 0, 1^dir, 0, dir );
        }
        if((fft_size<64)|((fft_size==64)&is_opt)){
            n=pnc*(qnc>>b);
            cuda_kernel_sgl( p_kernel, fft_size<64?((n+radix-1)/radix):n, 1, 1 );
        } else {
            cuda_kernel_sgl( p_kernel, pnc, qnc>>b, 1 );
        }
        if((!is_opt)|(!is_opt&(fft_size==64))|(fft_size>64)|(fft_size==8)){
            cuda_kernel_sep_i32( p_kernel, 3, fnx );
            cuda_kernel_sep_i32( p_kernel, 4, fny );
            cuda_kernel_sep_i32( p_kernel, 5, pnc*fnx*fny );
            if(fft_size<64){				
                cuda_kernel_sep_i32( p_kernel, 6, pnc );
            }
        }
    }
    
    {
        static const char ofs[]={4,5,5,6};
        int is_fuse=(mask>> 3)&0x1;
        int is_relu=(mask>>24)&0x1;
        int i=(is_fuse<<1)|is_relu;
        int o=ofs[dir==0?i:(2+is_relu)];
        p_kernel=&Op->kfft[2];
        idc_create_fft_kernel_c2r( p_kernel, p_ctx, axis, prc, dir, is_split, is_fuse, is_relu );
        n=q*n_cells_per_chl;
        cuda_kernel_sgl( p_kernel, c?((n+radix-1)/radix):bat, c?1:q, 1 );
        cuda_kernel_sep_i32( p_kernel, o++, onx );
        cuda_kernel_sep_i32( p_kernel, o++, ony );
        cuda_kernel_sep_i32( p_kernel, o++, ldo );
        if(c){
            cuda_kernel_sep_i32( p_kernel, o++, n_cells_per_chl );
            if(is_split){
                cuda_kernel_sep_i32( p_kernel, o++, n      );
                cuda_kernel_sep_i32( p_kernel, o++, grid_x );
                cuda_kernel_sep_i32( p_kernel, o++, grid_y );
                cuda_kernel_sep_i32( p_kernel, o++, dx     );
                cuda_kernel_sep_i32( p_kernel, o++, dy     );
            }
        }
    }
    n=((fft_size>>1)+1)*fft_size+(fft_size>8?0:8);
    idc_flatcgemm_create_kernel( &Op->kcgemm, p_ctx, prc, n, n_cells_per_chl, pnc, qnc, dir );
    cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(fft_size*fft_size)) );
    n<<=(prc!=2?3:2);
    Op->divpt[0]=n*n_cells_per_chl*p;
    Op->divpt[1]=n*pnc*(qnc>>b);
    return (Op->divpt[0]+Op->divpt[1]+n*n_cells_per_chl*q);
}
size_t idc_fftconv_createOp_grad( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, int prc, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat, int pad_x, int pad_y )
{
    int nx, ny, n, dx, dy, grid_x, grid_y, n_cells_per_chl, b, p, q, fft_size, axis, radix, is_split;
    cuda_kernel_t* p_kernel;
    
    nx=pnx+(pad_x<<1);
    ny=pny+(pad_y<<1);
    n=nx>ny?nx:ny;		
    if(n<=32)
    {
        fft_size=idc_minls(n);
        fft_size=fft_size<=8?8:fft_size;
        axis=idc_bhs(fft_size)-3;
        grid_x=grid_y=1;
        is_split=0;
    } 
    else 
    {
        fft_size=32;
        axis=2;
        dx=33-fnx;
        dy=33-fny;
        grid_x=(qnx+dx-1)/dx;
        grid_y=(qny+dy-1)/dy;
        is_split=1;
        if(n<=128){
            is_split=(idc_popc(grid_x)>1)|(idc_popc(grid_y)>1);
            if(is_split==0){
                fft_size=idc_minls(n);
                axis=idc_bhs(fft_size)-3;
                grid_x=grid_y=1;
            }
        }
    }
    n_cells_per_chl=bat*grid_x*grid_y;
    if((fft_size<=32)&(n_cells_per_chl>8)) return idc_cellconv_createOp_grad( Op, p_ctx, prc, fnx, fny, pnx, pny, pnc, ldp, qnx, qny, qnc, ldq, bat, pad_x, pad_y );
    b=prc&1;
    p=pnc>>b;
    q=qnc>>b;
    b=axis<3;
    radix=fft_size>8?8:16;
    is_split=(grid_x>1)|(grid_y>1);
    
    {
        int is_pad=(pad_x!=0)|(pad_y!=0);
        int is_ext=((pnx!=fft_size)|(pny!=fft_size))&(1^is_pad);
        p_kernel=&Op->kfft[0];
        idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, is_split, is_ext, is_pad, 0 );
        n=p*n_cells_per_chl;
        cuda_kernel_sgl( p_kernel, b?((n+radix-1)/radix):bat, b?1:p, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, pnx );
        cuda_kernel_sep_i32( p_kernel, 4, pny );
        cuda_kernel_sep_i32( p_kernel, 5, ldp );
        if(b){
            cuda_kernel_sep_i32( p_kernel, 6, n_cells_per_chl );
            if(is_split){
                cuda_kernel_sep_i32( p_kernel, 7, n      );
                cuda_kernel_sep_i32( p_kernel, 8, grid_x );
                cuda_kernel_sep_i32( p_kernel, 9, grid_y );
                cuda_kernel_sep_i32( p_kernel,10, dx     );
                cuda_kernel_sep_i32( p_kernel,11, dy     );
            }
        }
        if(is_pad){
            cuda_kernel_sep_i32( p_kernel, is_split?12:(axis<3?7:6), pad_x );
            cuda_kernel_sep_i32( p_kernel, is_split?13:(axis<3?8:7), pad_y );
        }
    }
    
    {
        p_kernel=&Op->kfft[1];
        idc_create_fft_kernel_r2c( p_kernel, p_ctx, axis, prc, is_split, 1, 0, 0 );		
        n=q*n_cells_per_chl;
        cuda_kernel_sgl( p_kernel, b?((n+radix-1)/radix):bat, b?1:q, 1 );		
        cuda_kernel_sep_i32( p_kernel, 3, qnx );
        cuda_kernel_sep_i32( p_kernel, 4, qny );
        cuda_kernel_sep_i32( p_kernel, 5, ldq );
        if(b){
            cuda_kernel_sep_i32( p_kernel, 6, n_cells_per_chl );
            if(is_split){
                cuda_kernel_sep_i32( p_kernel, 7, n      );
                cuda_kernel_sep_i32( p_kernel, 8, grid_x );
                cuda_kernel_sep_i32( p_kernel, 9, grid_y );
                cuda_kernel_sep_i32( p_kernel,10, dx     );
                cuda_kernel_sep_i32( p_kernel,11, dy     );
            }
        }
    }
    
    {
        p_kernel=&Op->kfft[2];		
        n=pnc*q;
        cuda_kernel_sgl( p_kernel, fft_size<64?((n+radix-1)/radix):n, 1, 1 );
        if(((fft_size>8)|(fft_size<128))&(((fnx==3)&(fny==3))|((fnx==5)&(fny==5)))){
            idc_create_fft_kernel_c2r_grad_opt( p_kernel, p_ctx, axis, prc, fnx>3 );
        } else {
            idc_create_fft_kernel_c2r_grad( p_kernel, p_ctx, axis, prc );		
            cuda_kernel_sep_i32( p_kernel, 4, fnx );
            cuda_kernel_sep_i32( p_kernel, 5, fny );
        }
    }
    
    n=((fft_size>>1)+1)*fft_size+(fft_size>8?0:8);
    idc_flatcgevv_create_kernel( &Op->kcgemm, p_ctx, prc, n, n_cells_per_chl, pnc, qnc );
    cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(bat*fft_size*fft_size)) );
    n<<=(prc!=2?3:2);
    Op->divpt[0]=n*n_cells_per_chl*p;
    Op->divpt[1]=n*n_cells_per_chl*q;
    return (Op->divpt[0]+Op->divpt[1]+n*pnc*q);
}
void idc_fftconv( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_bias, float alpha, CUstream s )
{
    CUdeviceptr d_a=d_aux;
    CUdeviceptr d_b=d_a+Op->divpt[0];
    CUdeviceptr d_c=d_b+Op->divpt[1];
    idc_fft2d_r2c( &Op->kfft[0], d_a, d_source, s );
    idc_fft2d_r2c( &Op->kfft[1], d_b, d_filter, s );
    idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
    idc_fft2d_c2r( &Op->kfft[2], d_target, d_c, d_bias, alpha, s );
}
void idc_fftconv_relu( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_x, float alpha, float slope, CUstream s )
{
    CUdeviceptr d_a=d_aux;
    CUdeviceptr d_b=d_a+Op->divpt[0];
    CUdeviceptr d_c=d_b+Op->divpt[1];
    idc_fft2d_r2c( &Op->kfft[0], d_a, d_source, s );
    idc_fft2d_r2c( &Op->kfft[1], d_b, d_filter, s );
    idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
    idc_fft2d_c2r_relu( &Op->kfft[2], d_target, d_c, d_x, alpha, slope, s );
}
void idc_fftconv_grad( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float ratio, CUstream s )
{
    CUdeviceptr d_a=d_aux;
    CUdeviceptr d_b=d_a+Op->divpt[0];
    CUdeviceptr d_c=d_b+Op->divpt[1];
    idc_fft2d_r2c( &Op->kfft[0], d_a, d_x, s );
    idc_fft2d_r2c( &Op->kfft[1], d_b, d_y, s );
    idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
    idc_fft2d_c2r( &Op->kfft[2], d_z, d_c, 0, ratio, s );
}