#ifndef __fft_argmask_h__
#define __fft_argmask_h__

#include "../../include/cuda/cuda_kernel.h"

#define FFT_AM_P_P_P_S				(AM(0,PA)|AM(1,PA)|AM(2,PA)|AM(3,SA))
#define FFT_AM_P_P_P_S_S			(AM(0,PA)|AM(1,PA)|AM(2,PA)|AM(3,SA)|AM(4,SA))
#define FFT_AM_P_P_P_S_S_S			(AM(0,PA)|AM(1,PA)|AM(2,PA)|AM(3,SA)|AM(4,SA)|AM(5,SA))
#define FFT_AM_P_P_P_S_S_S_S		(AM(0,PA)|AM(1,PA)|AM(2,PA)|AM(3,SA)|AM(4,SA)|AM(5,SA)|AM(6,SA))
#define FFT_AM_P_P_P_S_S_S_S_S		(AM(0,PA)|AM(1,PA)|AM(2,PA)|AM(3,SA)|AM(4,SA)|AM(5,SA)|AM(6,SA)|AM(7,SA))
#define FFT_AM_P_P_P_S_S_S_S_S_S	(AM(0,PA)|AM(1,PA)|AM(2,PA)|AM(3,SA)|AM(4,SA)|AM(5,SA)|AM(6,SA)|AM(7,SA)|AM(8,SA))

#endif