#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../common/common.h"
#include"../deepcore/deepcore.h"
#pragma comment( lib, "../deepcore/deepcore.lib" )
#pragma comment( lib, "cuda.lib" )

int main()
{
	if(dc_init()!=dc_success){
		printf( "error : deepcore init failed!\n" );
		exit(0);
	}
	dc_set_device(0);

	tensor_shape_t shapes[]={{48,3,32,32,1},{48,3,32,64,1},{24,3,64,128,1},{8,3,32,64,1},{8,3,64,128,1},{8,3,32,256,1},{6,3,64,256,1}};

	for( int dir=0; dir<2; ++dir )
	{
		for( int e=0; e<sizeof(shapes)/sizeof(shapes[0]); e++ )
		{
			int pn=shapes[e].ds;
			int fn=shapes[e].fs;
			int qn=pn-fn+1;
			int pnc=shapes[e].pnc;
			int qnc=shapes[e].qnc;
			int bat=shapes[e].bat;
			int stride=1;
			int inc=dir==0?pnc:qnc;
			int onc=dir==0?qnc:pnc;
			int in=dir==0?pn:qn;
			int on=dir==0?qn:pn;
			int pad=dir==0?0:(fn-1);
			uint64_t qshape=dc_create_tensor_shape( dcMaskPrecisionFloat, (qn<<16)|qn, (qnc<<16)|bat );
			uint64_t pshape=dc_create_tensor_shape( dcMaskPrecisionFloat, (pn<<16)|pn, (pnc<<16)|bat );
			uint64_t kshape=dc_create_tensor_shape_filter( dcMaskPrecisionFloat, (fn<<16)|fn, (qnc<<16)|pnc );
			uint64_t ishape=dir==0?pshape:qshape;
			uint64_t oshape=dir==0?qshape:pshape;

			dc_convOp Op;
		    size_t auxnb;
			if(dc_create_convOp( &Op, &auxnb, dcMaskPrecisionFloat|dir, 1, pshape, kshape, qshape, (stride<<8)|stride )!=dc_success){
				printf( "error : convOp create failed!\n" );
				dc_exit();
				return 0;
			}
			void *d_a, *d_b, *d_c, *d_aux;
			dc_create_tensor( (void**)&d_a, ishape );
			dc_create_tensor( (void**)&d_b, kshape );
			dc_create_tensor( (void**)&d_c, oshape );
		    if(auxnb>0){
		        uint64_t shape_linear=dc_create_tensor_shape_linear( auxnb );
				dc_create_tensor( (void**)&d_aux, shape_linear ); 
			}

			float* a=new float[bat*inc*in*in];
			float* b=new float[pnc*qnc*fn*fn];
			float* c=new float[bat*onc*on*on];
			float* d=new float[bat*onc*on*on];

			for( int i=0; i<inc; ++i )
			{
				for( int z=0; z<bat; ++z ){
					for( int y=0; y<in; ++y ){
						for( int x=0; x<in; ++x ){
							a[((i*bat+z)*in+y)*in+x]=((float)rand())/RAND_MAX;
						}
					}
				}
			}
			for( int i=0; i<fn*fn*pnc*qnc; ++i ){
				b[i]=((float)rand())/RAND_MAX;
			}

			for( int i=0; i<onc; ++i ){
				for( int s=0; s<bat; ++s ){
					conv( &c[(i*bat+s)*on*on], &a[s*in*in], &b[i*(dir==0?(pnc*fn*fn):(fn*fn))], dir, in, in, fn, fn, on, on, inc, bat, pad, pad, dir==0?(fn*fn):(pnc*fn*fn) );
				}
			}

			dc_tensor_store( d_a, ishape, a, bat*in*in*sizeof(float), bat*in*in*sizeof(float), inc, NULL );
			dc_tensor_store( d_b, kshape, b, pnc*fn*fn*sizeof(float), pnc*fn*fn*sizeof(float), qnc, NULL );
			if(dc_conv( Op, d_aux, d_c, d_a, d_b, NULL, 1.f, NULL )!=dc_success){
				printf( "error: conv exec failed!\n" );
			}
			dc_tensor_load( d, bat*on*on*sizeof(float), d_c, oshape, bat*on*on*sizeof(float), onc, NULL );
			cuCtxSynchronize();

			bool is_ok=check( c, d, onc*bat*on*on );
			if(!is_ok){
				printf( "examples[%d][%d] is compute failed!\n", dir, e );
			}
			dc_release_tensor( d_a );
			dc_release_tensor( d_b );
			dc_release_tensor( d_c );
		    if(auxnb!=0){
				dc_release_tensor( d_aux );
			}
			dc_destroy_convOp(Op);
			delete[] a;
			delete[] b;
			delete[] c;
			delete[] d;
			if(!is_ok) goto __EXIT__;
		}
	}
__EXIT__:
	dc_exit();
}