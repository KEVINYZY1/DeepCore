#ifndef __dnnCore_h__
#define __dnnCore_h__

#if defined(_MSC_VER)&&(defined(_WIN32)||defined(_WIN64))
	#if defined(_DLL)||defined(_WINDLL)
		#define __APIENTRY__ __declspec(dllexport)
	#else
		#define __APIENTRY__ __declspec(dllimport)
	#endif
	#if	defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
		#define __APICALL__
	#else
		#define __APICALL__ __stdcall
	#endif
#else
	#define __APIENTRY__
	#define __APICALL__
#endif

#ifndef uint64
	#ifdef _MSC_VER
		#define uint64 unsigned __int64
	#else
		#define uint64 unsigned long long
	#endif
#endif

#include<cuda.h>

#define dnnPrecisionMaskFloat		0x0
#define dnnPrecisionMaskMixed		0x1
#define dnnPrecisionMaskHalf		0x2

#define dnnTensorTypeMaskFilter		0x4
#define dnnTensorTypeMaskBias		0x5
#define dnnTensorTypeMaskFC		0x6

#define dnnPoolingMaskAvg		0x0
#define dnnPoolingMaskMax		0x4

#define	dnnActivationMaskReLU		0x0
#define	dnnActivationMaskPReLU		0x4
#define	dnnActivationMaskELU		0x5
#define	dnnActivationMaskTanh		0x6
#define	dnnActivationMaskSigmod		0x7

#define dnnReductionMaskAdd		0x0
#define dnnReductionMaskMax		0x4
#define dnnReductionMaskMin		0x5

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dnnConvOp	*	dnnConvOp;
typedef struct __dnnFFTConvOp	*	dnnFFTConvOp;
typedef struct __dnnCellConvOp	*	dnnCellConvOp;
typedef struct __dnnPoolingOp	*	dnnPoolingOp;
typedef struct __dnnBiasOp	*	dnnBiasOp;
typedef struct __dnnReductionOp	*	dnnReductionOp;
typedef struct __dnnActivationOp*	dnnActivationOp;

typedef enum dnnStatus{
	dnnSuccess=0		,
	dnnErrorInvalidValue	,
	dnnErrorInvalidDevice	,
	dnnErrorOutOfMaxSize	,
	dnnErrorOutOfMemory	,
	dnnErrorOutOfDeviceMemory
} dnnStatus_t;

__APIENTRY__	dnnStatus_t		__APICALL__ dnnInit();
__APIENTRY__	int			__APICALL__ dnnSuperdeviceCount();
__APIENTRY__	int			__APICALL__ dnnSuperdeviceSize( int );
__APIENTRY__	int			__APICALL__ dnnOptimalSuperdeviceID();
__APIENTRY__	void			__APICALL__ dnnSetDevice( int, int );

__APIENTRY__	uint64			__APICALL__ dnnCreateTensorShape( unsigned int, unsigned int, unsigned int, unsigned int, unsigned int );
__APIENTRY__	size_t			__APICALL__ dnnTensorPitch( uint64 );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateTensor( void**, uint64 );

__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateConvOp( dnnConvOp*, int, uint64, uint64, int, int );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateFFTConvOp( dnnFFTConvOp*, size_t*, int, unsigned int, uint64, uint64 );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateCellConvOp( dnnCellConvOp*, size_t*, int, unsigned int, uint64, uint64 );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreatePoolingOp( dnnPoolingOp*, int, unsigned int, uint64, int, int );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateBiasOp( dnnBiasOp*, int, unsigned int, uint64 );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateActivationOp( dnnActivationOp*, int, unsigned int );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateReductionOp( dnnReductionOp*, int, int, int, unsigned int );

__APIENTRY__	void			__APICALL__ dnnConv( dnnConvOp, void*, const void*, const void*, float, int, CUstream );
__APIENTRY__	void			__APICALL__ dnnFFTConv( dnnFFTConvOp, void*, void*, const void*, const void*, float, CUstream );
__APIENTRY__	void			__APICALL__ dnnCellConv( dnnCellConvOp, void*, void*, const void*, const void*, float, CUstream );
__APIENTRY__	void			__APICALL__ dnnPooling( dnnPoolingOp, void*, const void*, int, CUstream );
__APIENTRY__	void			__APICALL__ dnnUpdateBias( dnnBiasOp, void*, const void*, float, CUstream );
__APIENTRY__	void			__APICALL__ dnnActivate( dnnActivationOp, void*, const void*, const void*, uint64, float, float, int, CUstream );
__APIENTRY__	void			__APICALL__ dnnReduce( dnnReductionOp, void*, const void*, CUstream );

__APIENTRY__	void			__APICALL__ dnnDestroyConvOp( dnnConvOp );
__APIENTRY__	void			__APICALL__ dnnDestroyFFTConvOp( dnnFFTConvOp );
__APIENTRY__	void			__APICALL__ dnnDestroyCellConvOp( dnnCellConvOp );
__APIENTRY__	void			__APICALL__ dnnDestroyPoolingOp( dnnPoolingOp );
__APIENTRY__	void			__APICALL__ dnnDestroyBiasOp( dnnBiasOp );
__APIENTRY__	void			__APICALL__ dnnDestroyActivationOp( dnnActivationOp );
__APIENTRY__	void			__APICALL__ dnnDestroyReductionOp( dnnReductionOp );
__APIENTRY__	void			__APICALL__ dnnExit();

#ifdef __cplusplus
}
#endif

#endif
