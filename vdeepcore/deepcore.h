#ifndef __deepcore_h__
#define __deepcore_h__

#if defined(_MSC_VER)&&(defined(_WIN32)||defined(_WIN64))
	#ifdef _DLL
		#define DEEPCOREAPIENTRY __declspec(dllexport)
	#else
		#define DEEPCOREAPIENTRY __declspec(dllimport)
	#endif
#else
	#define DEEPCOREAPIENTRY
#endif

#include<stdint.h>
#include<cuda.h>

#define dcMaskDirectionForward  0x00
#define dcMaskDirectionBackward 0x01
#define dcMaskAddBiasOrMulDrv   0x08
#define	dcMaskActivationRelu    0x10

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dc_convOp    * dc_convOp;
typedef struct __dc_gemmOp    * dc_gemmOp;
typedef struct __dc_fftconvOp * dc_fftconvOp;
typedef struct __dc_biasOp    * dc_biasOp;
typedef struct __dc_bnormOp   * dc_bnormOp;

typedef enum dc_status{
	dc_success=0                 ,
	dc_error_invalid_device      ,
	dc_error_invalid_driver      ,
	dc_error_invalid_value       ,
	dc_error_no_active_device    ,
	dc_error_out_of_memory       ,
	dc_error_out_of_device_memory,
	dc_error_mutually_exclusive  ,
	dc_error_mismatch            ,
	dc_error_unsupported
} dc_status_t;

DEEPCOREAPIENTRY dc_status_t dc_init();
DEEPCOREAPIENTRY int         dc_get_device_count();
DEEPCOREAPIENTRY dc_status_t dc_set_device( int );

DEEPCOREAPIENTRY uint64_t    dc_create_tensor_shape( uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    dc_create_tensor_shape_input( uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    dc_create_tensor_shape_filter( uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    dc_create_tensor_shape_linear( size_t );
DEEPCOREAPIENTRY dc_status_t dc_create_tensor( void**, uint64_t );
DEEPCOREAPIENTRY dc_status_t dc_release_tensor( void* );

DEEPCOREAPIENTRY dc_status_t dc_tensor_zero( void*, uint64_t, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_tensor_subzero( void*, uint64_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_tensor_copy( void*, uint64_t, const void*, uint64_t, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_tensor_subcopy( void*, uint64_t, const void*, uint64_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_tensor_store( void*, uint64_t, const void*, size_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_tensor_load( void*, size_t, const void*, uint64_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_create_convOp( dc_convOp*, size_t*, uint32_t, int, uint64_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t dc_create_gemmOp( dc_gemmOp*, uint32_t, int, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t dc_create_gemmOp_grad( dc_gemmOp*, uint32_t, int, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t dc_create_fftconvOp( dc_fftconvOp*, size_t*, int, uint64_t, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t dc_create_biasOp( dc_biasOp*, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t dc_create_bnormOp( dc_bnormOp*, uint32_t, uint64_t, uint64_t );

DEEPCOREAPIENTRY dc_status_t dc_conv( dc_convOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_gemm( dc_convOp, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_gemm_grad( dc_convOp, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_fftconv_grad( dc_fftconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_bias_grad( dc_biasOp, void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_bnorm( dc_bnormOp, void*, void*, void*, const void*, const void*, const void*, CUstream );
DEEPCOREAPIENTRY dc_status_t dc_bnorm_grad( dc_bnormOp, void*, void*, void*, const void*, const void*, const void*, const void*, const void*, CUstream );

DEEPCOREAPIENTRY dc_status_t dc_destroy_convOp( dc_convOp );
DEEPCOREAPIENTRY dc_status_t dc_destroy_gemmOp( dc_gemmOp );
DEEPCOREAPIENTRY dc_status_t dc_destroy_fftconvOp( dc_fftconvOp );
DEEPCOREAPIENTRY dc_status_t dc_destroy_biasOp( dc_biasOp );
DEEPCOREAPIENTRY dc_status_t dc_destroy_bnormOp( dc_bnormOp );
DEEPCOREAPIENTRY dc_status_t dc_exit();

#ifdef __cplusplus
}
#endif

#endif
