DeepCore_v0.8

支持的硬件：计算能力为5.0,5.2,6.0,6.1的NVIDIA GPU。

DeepCore目前只支持一种数据格式：CNHW
    普通数据的布局为Channel_num*Batch_size*Height*Width
    对于filter数据的存储布局是QChannel_num*PChannel_num*Filter_size_y*Filter_size_x

目前channel_num必须是16的倍数,对于fftconv和cellconv,filter_size_x必须等于filter_size_y且必须>1

目前仅支持Relu内置激活函数

所有的测试均是和最新版本的cudnn6.0在GTX1080上做的对比，后续还会有更多的测试数据和更进一步的完善优化。
