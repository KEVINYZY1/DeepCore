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

	tensor_shape_t shapes[]={{48,3,32,64,8},{40,3,32,64,1},{24,3,32,64,32},{18,3,32,32,1},{10,3,32,32,32}};

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

			dc_fftconvOp Op;
			size_t auxnb;
			if(dc_create_fftconvOp( &Op, &auxnb, dcMaskPrecisionFloat|dir, pshape, kshape, qshape, 0 )!=dc_success){
				printf( "error : cellconvOp create failed!\n" );
				dc_exit();
				return 0;
			}

			void *d_a, *d_b, *d_c;
			CUdeviceptr auxbuf;
			dc_create_tensor( (void**)&d_a, ishape );
			dc_create_tensor( (void**)&d_b, kshape );
			dc_create_tensor( (void**)&d_c, oshape );
			cuMemAlloc( &auxbuf, auxnb );
			
			float* a=new float[bat*inc*in*in];
			float* b=new float[qnc*pnc*fn*fn];
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
					conv( &c[(i*bat+s)*on*on], &a[s*in*in], &b[i*pnc*fn*fn], dir, dir, in, in, fn, fn, on, on, inc, bat, pad, pad, fn*fn );
				}
			}

			dc_tensor_store( d_a, ishape, a, bat*in*in*sizeof(float), bat*in*in*sizeof(float), inc, NULL );
			dc_tensor_store( d_b, kshape, b, pnc*fn*fn*sizeof(float), pnc*fn*fn*sizeof(float), qnc, NULL );
			if(dc_fftconv( Op, (void*)auxbuf, d_c, d_a, d_b, NULL, 1.f, NULL )!=dc_success){
				printf( "error: conv exec failed!\n" );
			}
			dc_tensor_load( d, bat*on*on*sizeof(float), d_c, oshape, bat*on*on*sizeof(float), onc, NULL );
			cuCtxSynchronize();

			bool is_ok=check( c, d, bat*onc*on*on );
			if(!is_ok){
				printf( "examples[%d] is compute failed!\n", e );
				goto __LAB0;
			}

		__LAB0:
			dc_release_tensor( d_a );
			dc_release_tensor( d_b );
			dc_release_tensor( d_c );
			dc_destroy_fftconvOp(Op);
			cuMemFree(auxbuf);
			delete[] a;
			delete[] b;
			delete[] c;
			delete[] d;
			if(!is_ok) break;
		}
	}
	dc_exit();
}