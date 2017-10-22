#ifndef __deepcore_h__
#define __deepcore_h__

#if defined(_MSC_VER)&&(defined(_WIN32)||defined(_WIN64))
	#ifdef _DLL
		#define DEEPCOREAPIENTRY __declspec(dllexport)
	#else
		#define DEEPCOREAPIENTRY __declspec(dllimport)
	#endif
	#if	defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
		#define DEEPCOREAPICALL
	#else
		#define DEEPCOREAPICALL __stdcall
	#endif
#else
	#define DEEPCOREAPIENTRY
	#define DEEPCOREAPICALL
#endif

#include<stdint.h>
#include<cuda.h>

#define dcMaskDirectionForward      0x00000000
#define dcMaskDirectionBackward     0x00000001

#define dcMaskPrecisionFloat        0x00000000
#define dcMaskPrecisionHalf         0x00000002
#define dcMaskPrecisionMixed        0x00000004

#define dcMaskAddBiasOrMulDiff      0x00000008

#define	dcMaskActivationRelu        0x01000000

#define dcMaskPoolingAvg            0x00000000
#define dcMaskPoolingMax            0x00000008

#define dcMaskReduceAdd             0x00000000
#define dcMaskReduceMax             0x00000008
#define dcMaskReduceMin             0x00000010
#define dcMaskReduceDpxx            0x00000018
#define dcMaskReduceDpxy            0x00000020

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dc_convOp      *  dc_convOp;
typedef struct __dc_fftconvOp	*  dc_fftconvOp;
typedef struct __dc_cellconvOp	*  dc_cellconvOp;
typedef struct __dc_gemmOp      *  dc_gemmOp;
typedef struct __dc_reduceOp    *  dc_reduceOp;
typedef struct __dc_bnormOp     *  dc_bnormOp;

typedef enum dc_status{
	dc_success=0                 ,
	dc_error_invalid_device      ,
	dc_error_invalid_driver      ,
	dc_error_invalid_value       ,
	dc_error_no_active_device    ,
	dc_error_out_of_memory       ,
	dc_error_out_of_device_memory,
	dc_error_out_of_range        ,
	dc_error_mutually_exclusive  ,
	dc_error_mismatch
} dc_status_t;

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_init();
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_device_count();
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_superdevice_count();
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_superdevice_size( int );
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_superdevice_device_id( int, int );
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_optimal_superdevice_id();
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_set_device( int );

DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape( int, uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape_filter( int, uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape_linear( int, int );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_tensor( void**, uint64_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_release_tensor( void* );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_zero( void*, uint64_t, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_copy( void*, uint64_t, const void*, uint64_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_store( void*, uint64_t, const void*, size_t, size_t, size_t, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_load( void*, size_t, const void*, uint64_t, size_t, size_t, CUstream );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_convOp( dc_convOp*, int, uint32_t, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp( dc_fftconvOp*, size_t*, uint32_t, uint64_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp_grad( dc_fftconvOp*, size_t*, uint32_t, uint64_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp( dc_cellconvOp*, size_t*, uint32_t, uint64_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp_grad( dc_cellconvOp*, size_t*, uint32_t, uint64_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_gemmOp( dc_gemmOp*, int, uint32_t, uint64_t, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_reduceOp( dc_reduceOp*, uint32_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_bnormOp( dc_bnormOp*, uint32_t, uint64_t, uint64_t );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv( dc_convOp, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv_relu( dc_convOp, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv( dc_fftconvOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_relu( dc_fftconvOp, void*, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_grad( dc_fftconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv( dc_cellconvOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_relu( dc_cellconvOp, void*, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_grad( dc_cellconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_gemm( dc_gemmOp, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_gemm_relu( dc_gemmOp, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_gemm_grad( dc_gemmOp, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_reduce( dc_reduceOp, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_bnorm( dc_bnormOp, void*, void*, void*, const void*, const void*, const void*, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_bnorm_grad( dc_bnormOp, void*, void*, void*, const void*, const void*, const void*, const void*, const void*, CUstream );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_convOp( dc_convOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_fftconvOp( dc_fftconvOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_cellconvOp( dc_cellconvOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_gemmOp( dc_gemmOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_reduceOp( dc_reduceOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_bnormOp( dc_bnormOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_exit();

#ifdef __cplusplus
}
#endif

#endif
