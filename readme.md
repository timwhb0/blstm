# blstm-pytorch实现双人语音分离系统

这是一个基于pytorch实现的BLSTM的人声分离网络（本模型实现了双人声混合提取）。

## 语音预处理

准备数据可以使用preprocess文件夹中的脚本用于准备训练数据和验证数据，默认使用8k采样率的音频，如果使用16k音频请在downsample.py中进行降采样。mix_2speaker.py会混合两个音频，将长度对齐到最长的音频，并输出wav文件到指定的文件夹

## 语音特征提取
使用extract_feats.py进行特征提取，此步将会使用STFT算法对音频文件进行变换来提取特征。提取过后的特征将会用于模型的训练。
##模型
模型的基本结构：MLP encoder -> BLSTM ->MLP decoder

通过解码后得到的图像mask与输入相乘得到输出。

运行train.py文件将会训练模型并保存

## 损失函数
mag，acc，delta加权
## 效果

请参考result中的文件

## 参考资料

https://github.com/https://github.com/aishoot/LSTM_PIT_Speech_Separation.gitaishoot/LSTM_PIT_Speech_Separation.git

https://github.com/xuchenglin28/speech_separation.git    （基于此工程实现）

语音数据集来自 aishell
