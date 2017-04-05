DeepCore_v0.7

支持的硬件：计算能力为3.5,3.7,5.0,5.2,6.0,6.1的NVIDIA GPU(体验版期间只支持5.2和6.1)。

测试版仅支持单精度和混合精度，fp16的支持已完成约75%，不久后即可与大家见面。

DeepCore目前只支持一种数据格式：CNHW
    普通数据的布局为Channel_num*Batch_size*Height*Width
    对于filter数据的存储布局是QChannel_num*PChannel_num*Filter_size_y*Filter_size_x

目前channel_num必须是16的倍数,对于fftconv和cellconv,filter_size_x必须等于filter_size_y且必须>1

DeepCore目前仅支持正方形图像数据，也就是width=height (实际上内核层面没有这个限制，目前仅仅为了封装简单，后续会取消该限制）。

目前仅支持Relu内置激活函数。

所有的测试均是和最新版本的cudnn6.0在GTX1080上做的对比，后续还会有更多的测试数据和更进一步的完善优化。
