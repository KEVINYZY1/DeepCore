#ifndef __idc_tensor_shape_h__
#define __idc_tensor_shape_h__

#include"idc_bitop.h"
#include"idc_macro.h"

typedef struct idc_tensor_shape{
    uint32_t size;
    uint32_t pitch;
    uint32_t ncol;
} idc_tensor_shape_t;

typedef struct idc_op_param{
    uint32_t prc;
    uint32_t pnx;
    uint32_t pny;
    uint32_t pnc;
    uint32_t ldp;
    uint32_t qnx;
    uint32_t qny;	
    uint32_t qnc;
    uint32_t ldq;
    uint32_t bat;
    uint32_t fnx;
    uint32_t fny;
    uint32_t ldf;
} idc_op_param_t;

__forceinline void idc_get_tensor_shape( idc_tensor_shape_t* p_shape, uint64_t shape )
{
    uint32_t tt, prc, b, enb, nx, ny, bt, size, pitch, ncol;
    tt=((uint32_t)(shape>>56))&0x1;
    prc=(uint32_t)(shape>>62);
    b=prc&1;
    if(tt==0){
        nx=(((uint32_t)(shape>> 0))&0x01ff)+1;
        ny=(((uint32_t)(shape>> 9))&0x01ff)+1;
        bt=(((uint32_t)(shape>>18))&0x7fff)+1;
        size=bt*ny*nx;
        if(size<=32){
            pitch=idc_minls(size);
        } else
        if((size>32)&(size<=48)){
            pitch=IDC_AFFIS(size,16);
        } else
        if((size>64)&(size<=96)){
            pitch=IDC_AFFIS(size,32);
        } else 
        if(((size>48)&(size<=64))|((size>128)&(size<=192))){
            pitch=IDC_AFFIS(size,64);
        } else {
            pitch=IDC_AFFIS(size,128);
        }
        ncol=(((uint32_t)(shape>>33))&0x7fff)+1;
    } else
    if(tt==1){
        nx=(((uint32_t)(shape>> 0))&0x001f)+1;
        ny=(((uint32_t)(shape>> 5))&0x001f)+1;
        bt=(((uint32_t)(shape>>10))&0x7fff)+1;
        size=bt*ny*nx;
        pitch=IDC_AFFIS(size,8);
        ncol=(((uint32_t)(shape>>25))&0x7fff)+1;
    } else {
        size=(uint32_t)(shape>>b);
        pitch=size;
        ncol=b?2:1;
    }	
    enb=prc<2?4:2;
    p_shape->size=size*enb;
    p_shape->pitch=pitch*enb;
    p_shape->ncol=ncol>>b;
}

__forceinline void idc_get_op_param( idc_op_param_t* p_param, uint64_t shape_p, uint64_t shape_f, uint64_t shape_q )
{
    uint32_t size, pitch;
    p_param->prc=(uint32_t)(shape_p>>62);
    p_param->pnx=(((uint32_t)(shape_p>> 0))&0x01ff)+1;
    p_param->pny=(((uint32_t)(shape_p>> 9))&0x01ff)+1;
    p_param->pnc=(((uint32_t)(shape_p>>33))&0x7fff)+1;
    p_param->fnx=(((uint32_t)(shape_f>> 0))&0x001f)+1;
    p_param->fny=(((uint32_t)(shape_f>> 5))&0x001f)+1;
    p_param->qnx=(((uint32_t)(shape_q>> 0))&0x01ff)+1;
    p_param->qny=(((uint32_t)(shape_q>> 9))&0x01ff)+1;
    p_param->bat=(((uint32_t)(shape_q>>18))&0x7fff)+1;
    p_param->qnc=(((uint32_t)(shape_q>>33))&0x7fff)+1;
    size=p_param->bat*p_param->pny*p_param->pnx;
    if(size<=32){
        pitch=idc_minls(size);
    } else
    if((size>32)&(size<=48)){
        pitch=IDC_AFFIS(size,16);
    } else
    if((size>64)&(size<=96)){
        pitch=IDC_AFFIS(size,32);
    } else 
    if(((size>48)&(size<=64))|((size>128)&(size<=192))){
        pitch=IDC_AFFIS(size,64);
    } else {
        pitch=IDC_AFFIS(size,128);
    }
    p_param->ldp=pitch;
    size=p_param->bat*p_param->qny*p_param->qnx;
    if(size<=32){
        pitch=idc_minls(size);
    } else 
    if((size>32)&(size<=48)){
        pitch=IDC_AFFIS(size,16);
    } else
    if((size>64)&(size<=96)){
        pitch=IDC_AFFIS(size,32);
    } else 
    if(((size>48)&(size<=64))|((size>128)&(size<=192))){
        pitch=IDC_AFFIS(size,64);
    } else {
        pitch=IDC_AFFIS(size,128);
    }
    p_param->ldq=pitch;
    size=(p_param->pnc*p_param->fny*p_param->fnx)>>(p_param->prc&1);
    p_param->ldf=IDC_AFFIS(size,8);
}

#endif