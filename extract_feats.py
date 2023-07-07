import argparse
import multiprocessing
import os,sys
import numpy as np

from utils.audioread import audioread
from utils.sigproc import framesig,magspec
from utils.normhamming import normhamming
import time

def cal_phase_mag(filename):
    '''
    extract phase and feats for one utterance
    '''
    # audioread函数作用：读取音频文件，返回采样率和文件数据
    rate, sig, _ = audioread(filename)

    frames = framesig(sig, FLAGS.FFT_LEN, FLAGS.FRAME_SHIFT, lambda x: normhamming(x), True)
    phase, feats = magspec(frames, FLAGS.FFT_LEN)

    return phase, feats

def extract_mag_feats(item, mix_dir, clean1_dir, clean2_dir, mean_var_dict,total_train,total_labels1,total_labels2,feats_phase,labels1_phase,labels2_phase):

    # tfrecords to save the sequency consisting of feats and labels (optional for test)
    # tfrecords_name = os.path.join(FLAGS.output_dir, FLAGS.data_type, item.replace(".wav", ".tfrecords"))
    # writer = tf.python_io.TFRecordWriter(tfrecords_name)

    # extract feats for mixture
    phase_mix, feats = cal_phase_mag(os.path.join(mix_dir, item))

    # calculate intermediates for mean and variance, save to kaldi vector format
    mean_feats = np.sum(feats, 0)
    var_feats = np.sum(np.square(feats), 0)
    mean_var_dict[item] = str(np.shape(feats)[0])+'+'+' '.join(str(mean_feat) for mean_feat in mean_feats)+'+'+' '.join(str(var_feat) for var_feat in var_feats)

    # extract mag for clean as labels
    if clean1_dir != '' and clean2_dir != '':
        phase_clean1, labels1 = cal_phase_mag(os.path.join(clean1_dir, item.split(" ")[0]+'.wav'))
        phase_clean2, labels2 = cal_phase_mag(os.path.join(clean2_dir, item.split(" ")[1]))

        if FLAGS.apply_psm:
            labels1 = labels1 * np.cos(phase_mix-phase_clean1)
            labels2 = labels2 * np.cos(phase_mix-phase_clean2)
    else:
        labels1 = None
        labels2 = None
    
    total_train.append(feats)
    total_labels1.append(labels1)
    total_labels2.append(labels2)
    feats_phase.append(phase_mix)
    labels1_phase.append(phase_clean1)
    labels2_phase.append(phase_clean2)
    # write feats and labels into tfrecords
    # writer.write(make_sequence(feats, labels1, labels2).SerializeToString())

    return mean_var_dict

def cal_global_mean_std(filename, mean_var_dict):
    cmvn = np.zeros((2, int(FLAGS.FFT_LEN/2+1)), dtype=np.float32)
    frames = 0.0
    for line in mean_var_dict:
        tokens = line.strip().split('+')
        frames += float(tokens[0])
        utt_mean_tokens = tokens[1].strip().split()
        cmvn[0] += [np.float32(i) for i in utt_mean_tokens]
        utt_var_tokens = tokens[2].strip().split()
        cmvn[1] += [np.float32(i) for i in utt_var_tokens]

    mean = cmvn[0] / frames
    var = cmvn[1] / frames - mean ** 2
    var[var<=0] = 1.0e-20
    std = np.sqrt(var)

    print(mean)
    print(len(mean))
    print(std)
    print(len(std))
    np.savez(filename, mean_inputs=mean, stddev_inputs=std)

def main():
    print('Extract starts ...')
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
    feats=[]
    labels1=[]
    labels2=[]
    feats_phase=[]
    labels1_phase=[]
    labels2_phase=[]
    mix_dir = os.path.join(FLAGS.wav_dir, FLAGS.data_type, 'mix')
    if not os.path.exists(os.path.join(FLAGS.output_dir, FLAGS.data_type)):
        print(os.path.join(FLAGS.output_dir, FLAGS.data_type))
        os.makedirs(os.path.join(FLAGS.output_dir, FLAGS.data_type))

    if FLAGS.with_labels:
        clean1_dir = os.path.join(FLAGS.wav_dir, FLAGS.data_type, 's1')
        print(clean1_dir)
        clean2_dir = os.path.join(FLAGS.wav_dir, FLAGS.data_type, 's2')
    else:
        clean1_dir = ''
        clean2_dir = ''


    lists = [x for x in os.listdir(mix_dir) if x.endswith(".wav")]
    print(f"list length:{len(lists)}")
    print(lists)
    # check whether the cmvn file for training exist, remove if exist.
    if os.path.exists(FLAGS.inputs_cmvn):
        os.remove(FLAGS.inputs_cmvn)

    mean_vad_dict = multiprocessing.Manager().dict()
    total_mix=multiprocessing.Manager().list()
    total_labels1=multiprocessing.Manager().list()
    total_labels2=multiprocessing.Manager().list()
    feats_phase=multiprocessing.Manager().list()
    labels1_phase=multiprocessing.Manager().list()
    labels2_phase=multiprocessing.Manager().list()


    pool = multiprocessing.Pool(FLAGS.num_threads)
    workers = []
    for item in lists:
        workers.append(pool.apply_async(extract_mag_feats(item, mix_dir, clean1_dir, clean2_dir, mean_vad_dict,total_mix,total_labels1,total_labels2,feats_phase,labels1_phase,labels2_phase)))
    pool.close()
    pool.join()
    np.save("data\\tfrecords\\tr\\feats.npy",np.array(total_mix))
    np.save("data\\tfrecords\\tr\labels1",np.array(total_labels1))
    np.save("data\\tfrecords\\tr\labels2",np.array(total_labels2))
    np.save("data\\tfrecords\\tr\\feats_phase.npy",np.array(total_mix))
    np.save("data\\tfrecords\\tr\labels1_phase.npy",np.array(total_labels1))
    np.save("data\\tfrecords\\tr\labels2_phase.npy",np.array(total_labels2))
    # convert the utterance level intermediates for mean and var to global mean and std, then save
    cal_global_mean_std(FLAGS.inputs_cmvn, mean_vad_dict.values())
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
    print('Extract ends.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--with_labels',
        type=int,
        default=1,
        help='Whether extract features for the targets as labels, default to prepare labels.')
    parser.add_argument(
        '--data_type',
        type=str,
        default='tr',
        help='tr, cv, tt.')
    parser.add_argument(
        '--apply_psm',
        type=int,
        default=1,
        help='Whether use phase sensitive mask.')
    parser.add_argument(
        '--inputs_cmvn',
        type=str,
        default='\data\inputs_utts.cmvn',
        help='Path to save CMVN for the inputs'
    )
    parser.add_argument(
        '--wav_dir',
        type=str,
        default='data\wav',
        help='Directory to the input wav'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='\data\\tfrecords',
        help='Directory to save the features into tfrecords format'
    )
    parser.add_argument(
        '--FFT_LEN',
        type=int,
        default=512,
        help='The length of fft window.'
    )
    parser.add_argument(
        '--FRAME_SHIFT',
        type=int,
        default=256,
        help='The shift of samples when calculating fft.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=10,
        help='The number of threads to convert tfrecords files.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()

    