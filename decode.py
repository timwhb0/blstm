from utils.normhamming import normhamming
from utils.audioread import audioread
from utils.sigproc import framesig,magspec,deframesig
import numpy as np
import scipy.io.wavfile as wav
import argparse
import os
import sys
from model.model import Model
import scipy.signal as signal
from model.dataloader import AudioDataset,collate_fn
import torch.utils.data as Data
import soundfile as sf
FLAGS = None


def reconstruct(enhan_spec,phase):

    rate, nb_bits=8000,16
    nyquist_freq = 8000 / 2.0
    cutoff_freq = 2300
    order = 5
    b, a = signal.butter(order, cutoff_freq / nyquist_freq, btype='highpass')
    # 对音频信号进行滤波
    enhan_spec = signal.filtfilt(b, a, enhan_spec)
    spec_comp = enhan_spec*np.exp(1j*phase)
    enhan_frames = np.fft.irfft(spec_comp)
    enhan_sig = deframesig(enhan_frames.squeeze(0), 0, 512, 256, lambda x: normhamming(x))
    enhan_sig = enhan_sig / np.max(np.abs(enhan_sig)) 
    enhan_sig = enhan_sig * float(2 ** (nb_bits - 1))
    if nb_bits == 16:
        enhan_sig = enhan_sig.astype(np.int16)
    elif nb_bits == 32:
        enhan_sig = enhan_sig.astype(np.int32)
    return enhan_sig, rate


import torch
import torchaudio
import numpy as np
import os

def decode():
    # tfrecords_list, num_batches = read_list(FLAGS.lists_dir, FLAGS.data_type, FLAGS.batch_size)

    device = torch.device('cpu')
    # cmvn = np.load(FLAGS.inputs_cmvn)
    # inputs, inputs_cmvn, labels1, labels2, lengths = prepare_data(tfrecords_list, FLAGS.batch_size, 
    #                                                                 transform, cmvn, with_labels=True)

    model = Model(config,device)
    model.to(device)

    model.load_state_dict(torch.load("data/model/model.pth"))

    i=0
    for inputs,labels1,labels2 in train_loader:
        lengths=torch.tensor([j.shape[0] for j in inputs])
        print(len(lengths))
        inputs=inputs[0]
        labels1=labels1[0]
        labels2=labels2[0]
        sep1, sep2 = model.forward(inputs,lengths,labels1,labels2)
        sep1, sep2 = sep1.cpu().detach().numpy(), sep2.cpu().detach().numpy()

        
        enhan_sig1, rate = reconstruct(sep1,inputs_phase[i])
        enhan_sig2, rate = reconstruct(sep2,inputs_phase[i])

        
        savepath1 = os.path.join("output_1", f"signal{i}" + '_1.wav')
        savepath2 = os.path.join("output_1", f"signal{i}" + '_2.wav')

        
        sf.write(savepath1, enhan_sig1, rate, subtype='PCM_16')
        sf.write(savepath2, enhan_sig2, rate, subtype='PCM_16')

        print(f"Number of batch processed: {i+1}.")
        i=i+1


    print("Finished processing all batches.")

if __name__ =="__main__":
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
        default=1,
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
    param_num=30
    inputs=np.load("data\\tfrecords_1\\tr\\feats.npy",allow_pickle=True)[:param_num]
    inputs_phase=np.load("data\\tfrecords_1\\tr\\feats_phase.npy",allow_pickle=True)[:param_num]
    labels1=np.load("data\\tfrecords_1\\tr\\feats_phase.npy",allow_pickle=True)[:param_num]
    labels2=np.load("data\\tfrecords_1\\tr\\feats_phase.npy",allow_pickle=True)[:param_num]
    # labels1=np.load("data\\tfrecords\\tr\\labels1.npy",allow_pickle=True)[:param_num]
    # labels2=np.load("data\\tfrecords\\tr\\labels2.npy",allow_pickle=True)[:param_num]
    eval_dataset=AudioDataset(inputs,labels1,labels2)
    train_loader = Data.DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    


    decode()