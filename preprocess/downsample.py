import os
import argparse
import librosa
import soundfile as sf
def downsample_wav(input_path, output_path):
    # 读取音频文件
    y, sr = librosa.load(input_path, sr=16000)
    # 降采样
    y_ds = librosa.resample(y,orig_sr=sr,target_sr=8000)
    # 保存音频文件
    sf.write(output_path, y_ds, 8000, subtype='PCM_16')

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Downsample wav files from 16k to 8k')
    parser.add_argument('input_dir', help='input directory containing wav files')
    parser.add_argument('output_dir', help='output directory for downsampled wav files')
    args = parser.parse_args()

    # 检查输入文件夹是否存在
    if not os.path.isdir(args.input_dir):
        print(f'Error: input directory {args.input_dir} does not exist')
        exit(1)

    # 检查输出文件夹是否存在
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    print(librosa.__version__)

    # 遍历输入文件夹中的所有 wav 文件
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.wav'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            # 降采样并保存输出文件
            downsample_wav(input_path, output_path)
            print(f'Successfully downsampled {input_path} to {output_path}')