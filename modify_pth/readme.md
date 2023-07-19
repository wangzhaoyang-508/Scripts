openmmlab推出了mmpretrain，我在mmpretrain上使用MAE方法预训练出来的.pth文件不能直接用于下游任务模型。

我在使用detrex中的VitDet时出现了类似问题。

此脚本将.pth文件中的vit_base骨干部分转化为detrex可识别的pretrain文件。

