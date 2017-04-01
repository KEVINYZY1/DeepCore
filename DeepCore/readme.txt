DeepCore_v0.7

支持的硬件：计算能力为3.5,3.7,5.0,5.2,6.0,6.1的NVIDIA GPU(测试版期间暂不支持3.5,3.7)。

测试版仅支持单精度和混合精度，fp16的支持已完成约75%，不久后即可与大家见面。

DeepCore只支持一种数据格式：CNHW :
    普通数据的布局为Channels*Batch_size*Height*Width
    对于filter数据的存储布局是QChannels*PChannels*Filter_size_y*Filter_size_x

DeepCore目前仅支持正方形图像数据，也就是width=height (实际上内核层面没有这个限制，目前仅仅为了封装简单，后续会取消该限制）。

目前仅支持Relu内置激活函数。

所有的测试均是和最新版本的cudnn5.1在GTX1080上做的对比，后续还会有更多的测试数据和更进一步的完善优化。
