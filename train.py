import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import argparse
import numpy as np
from utils.comp_dynamic_feature import comp_dynamic_feature
from model.model import Model
from model.dataloader import AudioDataset,collate_fn
import os.path
if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lists_dir',
        type=str,
        default='tmp/',
        help="List to show where the data is."
    )
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether the clean labels are included in the tfrecords.'
    )
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='tfrecords/tr_cmvn.npz',
        help="The global cmvn to normalize the inputs."
    )
    parser.add_argument(
        '--input_size',
        type=int,
        default=257,
        help="Input feature dimension (default 129 for 8kHz sampling rate)."
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=257,
        help="Output dimension (mask dimension, default 129 for 8kHz sampling rate)."
    )
    parser.add_argument(
        '--dense_layer',
        type=str,
        default='true',
        help="Whether to use dense layer on top of input layer, when grid lstm is applied, this parameter is set to false, otherwise, set to true."
    )
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=257*4,
        help="Number of units in a rnn layer."
    )
    parser.add_argument(
        '--rnn_num_layers',
        type=int,
        default=3,
        help="Number of rnn layers."
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default='relu',
        help="Mask avtivation funciton, now only support sigmoid or relu"
    )
    parser.add_argument(
        '--tflstm_size',
        type=int,
        default=0,
        help="unit size for grid lstm, 64"
    )
    parser.add_argument(
        '--tffeature_size',
        type=int,
        default=29,
        help="input size for the frequency dimension of grid lstm layer, 29"
    )
    parser.add_argument(
        '--tffrequency_skip',
        type=int,
        default=10,
        help="shift of the input for the frequency dimension of grid lstm layer, 10"
    )
    parser.add_argument(
        '--tflstm_layers',
        type=int,
        default=0,
        help="number of grid lstm layers, 1"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Minibatch size."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0005,
        help="Initial learning rate."
    )
    parser.add_argument(
        '--min_epochs',
        type=int,
        default=30,
        help="Minimum epochs when training the model."
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=10,
        help="Maximum epochs when training the model."
    )
    parser.add_argument(
        '--lr_reduction_factor',
        type=float,
        default=0.5,
        help="Factor for reducing the learning rate."
    )
    parser.add_argument(
        '--reduce_lr_threshold',
        type=float,
        default=0.0,
        help="Threshold to decide when to reduce the learning rate."
    )
    parser.add_argument(
        '--stop_threshold',
        type=float,
        default=0.0001,
        help="Threshold to decide when to stop training."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=12,
        help='Number of threads for paralleling.'
    )
    parser.add_argument(
        '--save_model_dir',
        type=str,
        default='data\model',
        help="Directory to save the training model in every epoch."
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.5,
        help="Keep probability for training with a dropout (default: 1-dropout_rate)."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.0,
        help="Normalize the max gradient."
    )
    parser.add_argument(
        '--del_factor',
        type=float,
        default=0.2,
        help="weight for delta objective function, if larger than 0, delta objective function is applied."
    )
    parser.add_argument(
        '--acc_factor',
        type=float,
        default=0.4,
        help="weight for acceleration objective function, if larger than 0, delta objective function is applied."
    )
    parser.add_argument(
        '--dynamic_win',
        type=int,
        default=2,
        help="window size of the order in calculation of dynamic objective functions, default is 2."
    )
    parser.add_argument(
        '--mag_factor',
        type=float,
        default=0.4,
        help="weight for static (i.e., magnitude) objective function, if larger than 0, static objective function is applied."
    )
    parser.add_argument(
        '--tPSA',
        type=int,
        default=0,
        help="Whether use truncted PSA."
    )
    parser.add_argument(
        '--power_num',
        type=int,
        default=2,
        help="The power to calculate the loss, if set to 2, it's squared L2, if set to 1, it's L1."
    )
    FLAGS, unparsed = parser.parse_known_args()
    config=FLAGS
    train_inputs=np.load('data\\tfrecords\\tr\\feats.npy',allow_pickle=True)
    train_labels1=np.load('data\\tfrecords\\tr\\labels1.npy',allow_pickle=True)
    train_labels2=np.load('data\\tfrecords\\tr\\labels2.npy',allow_pickle=True)
    # 创建数据集和数据加载器
    train_set = AudioDataset(train_inputs, train_labels1, train_labels2)

    train_loader = Data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, collate_fn=collate_fn)

     #寻找cuda
   
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用GPU
    else:
        device = torch.device("cpu")  # 使用CPU

    # 创建模型和优化器
    model = Model(FLAGS,device)
    model.requires_grad_(True)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # 定义损失函数
    criterion=model.cal_loss

   

    #使用GPU训练
    model.to(device)
    epoch_num=100
    # 训练循环
    for epoch in range(epoch_num):
        # 遍历训练集
        for mix, labels1, labels2 in train_loader:
           
            print(nn.utils.rnn.pad_sequence(mix,batch_first=True,padding_value=0).size())
            # 清零梯度并计算模型输出
            optimizer.zero_grad()
            length=torch.tensor([i.size()[0] for i in mix])
            model.forward(mix,length,labels1,labels2)

            # 计算损失并反向传播
            loss=model.cal_loss()
            print(f"epoch:{epoch},loss={loss}")
            model.loss.requires_grad_(True)
            model.loss.backward()
            optimizer.step()


    torch.save(model.state_dict(),os.path.join(FLAGS.save_model_dir,'model.pth'))







    

