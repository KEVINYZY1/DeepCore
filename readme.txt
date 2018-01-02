deepcore_v0.9

    DeepCore是一款超轻量级专为CNN批量训练量身打造的高度优化的核心计算库。

    支持的硬件：计算能力为5.0,5.2,6.0,6.1的NVIDIA GPU。

    支持单精度和混合精度（双字节存储，单精度计算）。

    DeepCore目前只支持一种数据格式：CNHW
    普通数据的布局为Channel_num*Batch_size*Height*Width;
    对于filter数据的存储布局是QChannel_num*PChannel_num*Filter_size_y*Filter_size_x。

    卷积操作目前支持三种算法：conv, fftconv, cellconv;
    支持分组卷积；
    对于fftconv和cellconv,filter_size_x和filter_size_y必须>1;
    通过dc_gemmOp支持1x1卷积。

    目前仅支持relu内置激活函数,forward支持relu激活函数,bias融合;backward支持relu求导融合以及其它任意激活函数的导数相乘融合。

    支持reduction操作。

    支持batch-normalization。


    vdeepcore是专门针对volta优化的版本且仅支持volta GPU,由于专门针对tensor core进行了优化，因此数据结构差别很大，因此为简单以及避免代码过度膨胀，从volta开始会有一个新的分支版本且与之前的版本不兼容
    
