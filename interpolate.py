import soundfile as sf
import numpy 
import os
from tqdm import tqdm
import scipy.signal as signal
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
 
	else:
		print("---  There is this folder!  ---")
def interpolate(filepath,output_path,frame_size=320, re_sr=16000):
    print(filepath)
    re_wav,fs = sf.read(filepath)
    f,t,nd = signal.stft(re_wav,fs = re_sr , nperseg = frame_size,noverlap=0,nfft=None,detrend=False,return_onesided=True,boundary='zeros',padded=True,axis=-1)
    # print(nd[0][200])
    nframes = len(re_wav) // frame_size
    #print(nd.shape)
    #print(nframes)
    left_frame = nd[:,0]
    right_frame = nd[:,0]
    right_num = 0
    left_num = 0
    for i in range(1,nframes):
        #print(abs(nd[:,i]))
        if sum(abs(nd[:,i])) != 0:
            left_frame = nd[:,i]
            left_num = i+1
            #print('left_num:',left_num)
            
        else:
            for j in range(i+1,nframes):
                if sum(abs(nd[:,j])) != 0:
                    right_frame = nd[:,j]
                    right_num = j - 1
                    break
                if j == nframes-1:
                    right_frame = nd[:,j]
                    right_num = j - 1
                #print('right:',right_num)
            for t in range(left_num,right_num+1):
                r_t = (t-left_num+1)/(2+right_num-left_num)
                nd[:,t] = r_t*right_frame+(1-r_t)*left_frame
        #print(abs(nd[:,i]))
        
    t,output_wav = signal.istft(nd, fs=re_sr, window='hann', nperseg=None, noverlap=0, nfft=None, input_onesided=True, boundary='zeros', time_axis=- 1, freq_axis=- 2)
    output_wav = output_wav[0:len(re_wav)]
    #print(output_wav.shape)
    sf.write(output_path, output_wav, samplerate=re_sr)
def create(root,l_rate):
    interloss_paths = os.path.join(root, 'interloss/'+str(l_rate)+'_loss/noisy/noisy_trainset_wav/')
    if not os.path.exists(interloss_paths):
        mkdir(interloss_paths)
    loss_paths = os.path.join(root, ''+str(l_rate)+'_loss/noisy/noisy_trainset_wav/')
    loss_files = os.listdir(loss_paths)
    for file in tqdm(loss_files):
        print(file)
        interpolate(os.path.join(loss_paths, file),os.path.join(interloss_paths,file))
root = '/E/zhangjinghong/TFGAN-PLC-main/data_m/'
l_rate = 50
create(root,l_rate)