#ifndef __platform_h__
#define __platform_h__

#include<cuda.h>
#include"../idc_macro.h"
#include"../idc_string.h"

typedef struct cuda_platform{	
    CUdevice devices   [IDC_MAX_DEVICES_PER_NODE];
    int      clock_rate[IDC_MAX_DEVICES_PER_NODE];
    int      nSM       [IDC_MAX_DEVICES_PER_NODE];
    int      sarch     [IDC_MAX_DEVICES_PER_NODE];
    int      slist     [IDC_MAX_DEVICES_PER_NODE+1];
    int      n_devices;
    int      n_sdevices;	
    int      opt_sdev_id;
} cuda_platform_t;

int cuda_platform_init( cuda_platform_t* );

#endif
