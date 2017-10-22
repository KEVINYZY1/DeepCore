#include"../../include/idc_bitop.h"
#include"../../include/conv/fftconv.h"

size_t idc_cellconv_createOp( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, uint32_t mask, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat, int pad_x, int pad_y )
{
    int dir, prc, b, inc, onc, p, q, snx, sny, lds, onx, ony, ldo, nx, ny, dx, dy, grid_x, grid_y, cell_size, n, axis, n_cells_per_chl, lda, ldb, is_split, perm_id;
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
    cell_size=n<=8?8:(n<=16?16:32);
    dx=cell_size-fnx+1;
    dy=cell_size-fny+1;
    grid_x=(onx+dx-1)/dx;
    grid_y=(ony+dy-1)/dy;
    n_cells_per_chl=bat*grid_x*grid_y;
    if((n_cells_per_chl>1)&(n_cells_per_chl<=8))
        return idc_fftconv_createOp( Op, p_ctx, mask, pnx, pny, pnc, ldp, fnx, fny, qnx, qny, qnc, ldq, bat, pad_x, pad_y );
    prc=(mask>>1)&0x3;
    b=prc&1;
    inc=dir?qnc:pnc;
    onc=dir?pnc:qnc;
    p=inc>>b;
    q=onc>>b;
    lda=(n_cells_per_chl>1)?n_cells_per_chl:p;
    ldb=dir?onc:q;
    lda=IDC_AFFIS(lda,32);
    ldb=IDC_AFFIS(ldb,32);
    axis=(cell_size>8)+(cell_size>16);
    is_split=(grid_x>1)|(grid_y>1);
    perm_id=is_split?2:(bat>1);
    
    {
        int is_pad=(pad_x!=0)|(pad_y!=0);
        int is_ext=((snx!=cell_size)|(sny!=cell_size))&(1^is_pad);
        p_kernel=&Op->kfft[0];
        idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, perm_id, is_ext, is_pad, 0 );
        cuda_kernel_sep_i32( p_kernel, 3, snx );
        cuda_kernel_sep_i32( p_kernel, 4, sny );			
        cuda_kernel_sep_i32( p_kernel, 5, lda );
        cuda_kernel_sep_i32( p_kernel, 6, lds );
        n=n_cells_per_chl>1?n_cells_per_chl:p;
        cuda_kernel_sgl( p_kernel, (n+15)>>4, n_cells_per_chl>1?p:1, 1 );
        if(is_split==0){	
            if((bat>1)&(is_pad==0)){ cuda_kernel_sep_i32( p_kernel, 7, snx*sny ); }
        } else {
            cuda_kernel_sep_i32( p_kernel, 7, n_cells_per_chl );
            cuda_kernel_sep_i32( p_kernel, 8, grid_x	      );
            cuda_kernel_sep_i32( p_kernel, 9, grid_y	      );
            cuda_kernel_sep_i32( p_kernel,10, dx              );
            cuda_kernel_sep_i32( p_kernel,11, dy              );
            cuda_kernel_sep_i32( p_kernel,12, 0               );
        }			
        if(is_pad){
            cuda_kernel_sep_i32( p_kernel, 7+6*is_split, pad_x );
            cuda_kernel_sep_i32( p_kernel, 8+6*is_split, pad_y );
        }
    }
    
    {
        int fxy=fnx*fny;
        int ldr=pnc*fxy;
        int is_opt=((fnx==3)&(fny==3))|((fnx==5)&(fny==5))|((fnx==7)&(fny==7));
        p_kernel=&Op->kfft[1];
        if((cell_size>8)&is_opt){
            idc_create_cellfft_kernel_r2c_opt( p_kernel, p_ctx, axis, prc, dir, (fnx>3)+(fnx>5) );
            cuda_kernel_sep_i32( p_kernel, 3, ldb );
            cuda_kernel_sep_i32( p_kernel, 4, ldr );
        } else {
            idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, 1, 1^dir, 0, dir );
            cuda_kernel_sep_i32( p_kernel, 3, fnx	      );
            cuda_kernel_sep_i32( p_kernel, 4, fny	      );
            cuda_kernel_sep_i32( p_kernel, 5, ldb	      );
            cuda_kernel_sep_i32( p_kernel, 6, dir?ldr:fxy );
            cuda_kernel_sep_i32( p_kernel, 7, dir?fxy:ldr );
        }
        cuda_kernel_sgl( p_kernel, ((dir?onc:q)+15)>>4, dir?p:inc, 1 );
    }
    
    {
        static const char ofs[]={4,5,5,6};
        int is_fuse=(mask>> 3)&0x1;
        int is_relu=(mask>>24)&0x1;
        int i=(is_fuse<<1)|is_relu;
        int o=ofs[dir==0?i:(i==0?0:(2+is_relu))];
        int radix=cell_size>16?8:16;
        n=n_cells_per_chl>1?n_cells_per_chl:q;
        p_kernel=&Op->kfft[2];
        idc_create_cellfft_kernel_c2r( p_kernel, p_ctx, axis, prc, dir, perm_id, is_fuse, is_relu );
        cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, n_cells_per_chl>1?q:1, 1 );		
        cuda_kernel_sep_i32( p_kernel, o++, onx );
        cuda_kernel_sep_i32( p_kernel, o++, ony );
        cuda_kernel_sep_i32( p_kernel, o++, ldo );
        cuda_kernel_sep_i32( p_kernel, o++, n_cells_per_chl>1?lda:ldb );
        if(is_split){
            cuda_kernel_sep_i32( p_kernel, o++, n      );
            cuda_kernel_sep_i32( p_kernel, o++, grid_x );
            cuda_kernel_sep_i32( p_kernel, o++, grid_y );
            cuda_kernel_sep_i32( p_kernel, o++, dx     );
            cuda_kernel_sep_i32( p_kernel, o++, dy     );
        }
    }
    
    b=prc!=2?3:2;
    n=((cell_size>>1)+1)*cell_size;
    p_kernel=&Op->kcgemm;
    if(n_cells_per_chl>1){
        idc_cgemm_create_kernel( p_kernel, p_ctx, dir, prc, n, n_cells_per_chl, inc, onc, lda<<=b, ldb<<=b );
    } else {
        idc_cgemv_create_kernel( p_kernel, p_ctx, dir, prc, n, inc, onc, lda, ldb, ldb );
        n<<=b;
    }
    cuda_kernel_sep_f32( p_kernel, 3, (float)(1.0/(cell_size*cell_size)) );
    Op->divpt[0]=n*(n_cells_per_chl>1?p:1)*lda;
    Op->divpt[1]=n*(dir?p:inc)*ldb;
    return (Op->divpt[0]+Op->divpt[1]+n*(n_cells_per_chl>1?(q*lda):ldb));
}
size_t idc_cellconv_createOp_grad( idc_fftconvOp_t* Op, const cuda_context_t* p_ctx, int prc, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int qnx, int qny, int qnc, int ldq, int bat, int pad_x, int pad_y )
{
    int b, p, q, cell_size, nx, ny, n, dx, dy, grid_x, grid_y, n_cells_per_chl, lda, ldb, axis, is_split, perm_id;
    cuda_kernel_t* p_kernel;
    
    nx=pnx+(pad_x<<1);
    ny=pny+(pad_y<<1);
    n=nx>ny?nx:ny;
    cell_size=n<=8?8:(n<=16?16:32);
    dx=cell_size-fnx+1;
    dy=cell_size-fny+1;
    grid_x=(qnx+dx-1)/dx;
    grid_y=(qny+dy-1)/dy;
    n_cells_per_chl=bat*grid_x*grid_y;
    if(n_cells_per_chl<8)
        return idc_fftconv_createOp_grad( Op, p_ctx, prc, pnx, pny, pnc, ldp, fnx, fny, qnx, qny, qnc, ldq, bat, pad_x, pad_y );
    b=prc&1;	
    p=pnc>>b;
    q=qnc>>b;
    lda=IDC_AFFIS(p,32);
    ldb=IDC_AFFIS(q,32);
    axis=(cell_size>8)+(cell_size>16);
    is_split=(grid_x>1)|(grid_y>1);
    perm_id=is_split?2:1;
    
    {
        int is_pad=(pad_x!=0)|(pad_y!=0);
        int is_ext=((pnx!=cell_size)|(pny!=cell_size))&(1^is_pad);
        p_kernel=&Op->kfft[0];
        idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, perm_id, is_ext, is_pad, 0 );
        n=is_split?p:n_cells_per_chl;
        cuda_kernel_sgl( p_kernel, (n+15)>>4, is_split?n_cells_per_chl:p, 1 );
        cuda_kernel_sep_i32( p_kernel, 3, pnx );
        cuda_kernel_sep_i32( p_kernel, 4, pny );
        cuda_kernel_sep_i32( p_kernel, 5, lda );
        cuda_kernel_sep_i32( p_kernel, 6, ldp );
        if(is_split==0){		
            if(is_pad==0){ cuda_kernel_sep_i32( p_kernel, 7, pnx*pny ); }
        } else {
            cuda_kernel_sep_i32( p_kernel, 7, n      );
            cuda_kernel_sep_i32( p_kernel, 8, grid_x );
            cuda_kernel_sep_i32( p_kernel, 9, grid_y );
            cuda_kernel_sep_i32( p_kernel,10, dx     );
            cuda_kernel_sep_i32( p_kernel,11, dy     );
            cuda_kernel_sep_i32( p_kernel,12, 1      );
        }
        if(is_pad){
            cuda_kernel_sep_i32( p_kernel, 7+6*is_split, pad_x );
            cuda_kernel_sep_i32( p_kernel, 8+6*is_split, pad_y );
        }
    }
    
    {
        p_kernel=&Op->kfft[1];
        idc_create_cellfft_kernel_r2c( p_kernel, p_ctx, axis, prc, perm_id, 1, 0, 0 );
        n=is_split?q:n_cells_per_chl;
        cuda_kernel_sgl( p_kernel, (n+15)>>4, is_split?n_cells_per_chl:q, 1 );		
        cuda_kernel_sep_i32( p_kernel, 3, qnx );
        cuda_kernel_sep_i32( p_kernel, 4, qny );			
        cuda_kernel_sep_i32( p_kernel, 5, ldb );
        cuda_kernel_sep_i32( p_kernel, 6, ldq );
        cuda_kernel_sep_i32( p_kernel, 7, is_split?n:(qnx*qny) );
        if(is_split){
            cuda_kernel_sep_i32( p_kernel, 8, grid_x );
            cuda_kernel_sep_i32( p_kernel, 9, grid_y );
            cuda_kernel_sep_i32( p_kernel,10, dx     );
            cuda_kernel_sep_i32( p_kernel,11, dy     );
        }
    }
    
    {
        int radix=cell_size>16?8:16;
        int o=4;
        p_kernel=&Op->kfft[2];
        if((cell_size>8)&(((fnx==3)&(fny==3))|((fnx==5)&(fny==5)))){
            idc_create_cellfft_kernel_c2r_grad_opt( p_kernel, p_ctx, axis, prc, fnx>3 );
        } else {
            idc_create_cellfft_kernel_c2r_grad( p_kernel, p_ctx, axis, prc );
            cuda_kernel_sep_i32( p_kernel, 4, fnx );
            cuda_kernel_sep_i32( p_kernel, 5, fny );
            o=6;
        }
        cuda_kernel_sgl( p_kernel, (pnc+radix-1)/radix, q, 1 );
        cuda_kernel_sep_i32( p_kernel, o++, pnc*fnx*fny );
        cuda_kernel_sep_i32( p_kernel, o++, lda         );		
    }
    
    b=prc!=2?3:2;
    p_kernel=&Op->kcgemm;
    n=((cell_size>>1)+1)*cell_size;
    idc_cgemm_create_kernel( p_kernel, p_ctx, 2, prc, n, pnc, n_cells_per_chl, qnc, lda<<=b, ldb<<=b );
    cuda_kernel_sep_f32( p_kernel, 3, (float)(1.0/(bat*cell_size*cell_size)) );
    Op->divpt[0]=n*n_cells_per_chl*lda;
    Op->divpt[1]=n*n_cells_per_chl*ldb;
    return (Op->divpt[0]+Op->divpt[1]+n*lda*q);
}
void idc_cellconv( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_x, float alpha, CUstream s )
{
    CUdeviceptr d_a=d_aux;
    CUdeviceptr d_b=d_a+Op->divpt[0];
    CUdeviceptr d_c=d_b+Op->divpt[1];
    idc_fft2d_r2c( &Op->kfft[0], d_a, d_source, s );
    idc_fft2d_r2c( &Op->kfft[1], d_b, d_filter, s );
    idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
    idc_fft2d_c2r( &Op->kfft[2], d_target, d_c, d_x, alpha, s );
}
void idc_cellconv_relu( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_x, float alpha, float slope, CUstream s )
{
    CUdeviceptr d_a=d_aux;
    CUdeviceptr d_b=d_a+Op->divpt[0];
    CUdeviceptr d_c=d_b+Op->divpt[1];
    idc_fft2d_r2c( &Op->kfft[0], d_a, d_source, s );
    idc_fft2d_r2c( &Op->kfft[1], d_b, d_filter, s );
    idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
    idc_fft2d_c2r_relu( &Op->kfft[2], d_target, d_c, d_x, alpha, slope, s );
}
void idc_cellconv_grad( idc_fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float ratio, CUstream s )
{
    CUdeviceptr d_a=d_aux;
    CUdeviceptr d_b=d_a+Op->divpt[0];
    CUdeviceptr d_c=d_b+Op->divpt[1];	
    idc_fft2d_r2c( &Op->kfft[0], d_a, d_x, s );
    idc_fft2d_r2c( &Op->kfft[1], d_b, d_y, s );
    idc_cgemm( &Op->kcgemm, d_c, d_a, d_b, s );
    idc_fft2d_c2r( &Op->kfft[2], d_z, d_c, 0, ratio, s );
}