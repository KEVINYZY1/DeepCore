#ifndef __fft_calcRF_h__
#define __fft_calcRF_h__

#include<math.h>
#include<vector_types.h>
#include"fft_bop.h"

#define PI 3.1415926535897931e+0

__forceinline void fft_calcRF( float2* const p, int n, double rt )
{
	int i=0;
	do{ 
		p[i].x=(float)cos(i*rt*-PI);
		p[i].y=(float)sin(i*rt*-PI);
	}while((++i)<n);
}

#endif