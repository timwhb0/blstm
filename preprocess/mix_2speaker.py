
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import numpy as np
from math import ceil
import sounddevice as sd
import soundfile as sf
import os.path


trNum=10000
cvNum=20
ttNum=10
file1 = "..\S0010"
file2 = "..\S0252_mic"
file_s1 = "..\data\wav\\tr\s1"
file_s2 = "..\data\wav\\tr\s2"
fileout="..\data\wav\\tr\mix"
os.environ["TQDM_DISABLE"] = "True"
def delete_files_in_folder(folder_path):
    # 获取文件夹中的所有文件列表
    file_list = os.listdir(folder_path)
    
    # 遍历文件列表，逐个删除文件
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
        
    print("所有文件已删除")
def mix_2speaker(file1,file2,outpath,file_s1,file_s2):
    #pad each signal to the longer one and mix them
 

    samples1,sample_rate1= librosa.load(file1,sr=8000)
    samples2, sample_rate2 = librosa.load(file2,sr=8000)

    # Find length of longest signal
    maxlength = max(len(samples1),len(samples2))  # 53556

    # Pad each signal to the length of the longest signal
    samples1 = np.pad(samples1, (0, maxlength - len(samples1)), 'constant', constant_values=(0))
    samples2 = np.pad(samples2, (0, maxlength - len(samples2)), 'constant', constant_values=(0))

    # Combine series together
    mixed_series = samples1 + samples2

    extrapadding = ceil(len(mixed_series) / sample_rate1) * sample_rate1 - len(mixed_series)
    mixed_series = np.pad(mixed_series, (0, extrapadding), 'constant', constant_values=(0))
    samples1 = np.pad(samples1, (0,extrapadding), 'constant', constant_values=(0))
    samples2 = np.pad(samples2, (0,extrapadding), 'constant', constant_values=(0))

    fileout=os.path.join(outpath,os.path.basename(file1)[:-4]+" "+os.path.basename(file2)[:-4]+".wav")
    if os.path.exists(fileout):
        os.remove(fileout)
        print("File deleted successfully.")
    sample_rate = 8000
    sf.write(fileout, mixed_series, sample_rate, subtype='PCM_16')
    sf.write(os.path.join(file_s1,os.path.basename(file1)), samples1, sample_rate, subtype='PCM_16')
    sf.write(os.path.join(file_s2,os.path.basename(file2)), samples2, sample_rate, subtype='PCM_16')


if __name__ == '__main__':

    s1_path=[]
    s2_path=[]
    for root, dirs, files in os.walk(file1):
        for file in files:
            # 检查文件是否以.wav结尾
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                s1_path.append(file_path)
    for root, dirs, files in os.walk(file2):
        for file in files:
            # 检查文件是否以.wav结尾
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                s2_path.append(file_path)
    delete_files_in_folder(fileout)
    print(len(s1_path))
    for i in range(trNum if trNum<max(len(s1_path),len(s2_path)) else len(s1_path)):
        print(s1_path[i])
        # print(s2_path[i])
        print(len(s2_path))
        
        mix_2speaker(s1_path[i],s2_path[i],fileout,file_s1,file_s2)
        print(f"progress:{i+1} / {trNum}")
   