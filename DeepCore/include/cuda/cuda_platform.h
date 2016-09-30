#ifndef __platform_h__
#define __platform_h__

#include<memory.h>
#include<string.h>
#include<cuda.h>
#include"../dc_macro.h"

typedef struct cuda_platform{	
	CUdevice	devices		[MAX_DEVICES_PER_NODE];
	int			clock_rate	[MAX_DEVICES_PER_NODE];
	int			nSM			[MAX_DEVICES_PER_NODE];
	int			sarch		[MAX_DEVICES_PER_NODE];
	int			slist		[MAX_DEVICES_PER_NODE+1];
	int			n_devices;
	int			n_sdevices;	
	int			opt_sdev_id;
} cuda_platform_t;

int cuda_platform_init( cuda_platform_t* const );

#endif