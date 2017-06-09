DeepCore_v0.8

支持的硬件：计算能力为5.0,5.2,6.0,6.1的NVIDIA GPU。

支持单精度，混合精度和半精度

DeepCore目前只支持一种数据格式：CNHW
    普通数据的布局为Channel_num*Batch_size*Height*Width
    对于filter数据的存储布局是QChannel_num*PChannel_num*Filter_size_y*Filter_size_x

卷积操作目前支持三种算法：conv, fftconv, cellconv, conv算法暂时不支持反向卷积核padding操作
对于fftconv和cellconv,channel_num必须是32的倍数，filter_size_x和filter_size_y必须>1

目前仅支持Relu内置激活函数

目前还不支持1x1的卷积核，但不久后即可提供支持。


所有的测试均是和最新版本的cudnn6.0在GTX1080上做的对比，后续还会有更多的测试数据和更进一步的完善优化。
