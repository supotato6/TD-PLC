from pesq import pesq
import librosa
from pystoi import stoi
from functools import reduce
import math
import numpy as np
import json
import os

# print(datanames)
# ori_data,fs1= librosa.load('./p232_028.wav',sr=None) 
# enh_data,fs2= librosa.load('./ests_p232_028.wav',sr=None)
# noisy,fs3= librosa.load('./NO_p232_028.wav',sr=None)
# ori_data = librosa.resample(y=ori_data,orig_sr = 48000 , target_sr = 16000)
# noisy = librosa.resample(y=noisy,orig_sr = 48000 , target_sr = 16000)
noisy_trend_avgpesq = []
inpainted_trend_avgpesq = []
noisy_trend_maxpesq = []
noisy_trend_minpesq = []
inpainted_trend_maxpesq = []
inpainted_trend_minpesq = []


trend_lossrate = [i/10 for i in range(10,100)]

for i in range(1,2):
# path_ = path +"/"
  loss_rate = i*10
  path = "/E/zhangjinghong/TFGAN-PLC-main/outputs/40_loss/clean/"
  #path = "/E/zhangjinghong/TFGAN-PLC-main/data_m/"+str(loss_rate)+"_loss/clean/noisy_testset_wav/"
  #noisy_path = "/E/zhangjinghong/TFGAN-PLC-main/outputs/frn/"+str(loss_rate)+"_loss/noisy/"
  noisy_path = "/E/zhangjinghong/TFGAN-PLC-main/outputs/40_loss/noisy/"
  #inpainted_path =  "/E/zhangjinghong/TFGAN-PLC-main/outputs/"+str(loss_rate)+"_loss/inpainted/"
  inpainted_path = "/E/zhangjinghong/TFGAN-PLC-main/outputs/40_loss/inpainted/"
  datanames = os.listdir(path)
  # ori_data,fs1 = librosa.load('/E/zhangjinghong/TFGAN-PLC-main/data_m/test/clean_testset_wav/',sr=None) 
  # ori_data = ori_data[:-1]
  # print(len(ori_data))
  def get_pesq(ref, deg):
      return pesq(16000, ref, deg, 'nb')
  
  pesq_list=[]
  stoi_list=[]
  noisy_pesq_list=[]
  noisy_stoi_list=[]
  for dataname in datanames:
    if not dataname.endswith('.wav'):
      continue
    else:
      ori_data,fs1 = librosa.load(path+dataname,sr=None)
      ori_data = ori_data[:-1]
      #enh_data,fs2= librosa.load(inpainted_path+dataname[:-4]+".wav_synthesis"+dataname[-4:],sr=None)
      enh_data,fs2= librosa.load(inpainted_path+dataname,sr=None)
      enh_data = enh_data[:len(ori_data)]
      noisy_data,fs3 = librosa.load(noisy_path+dataname,sr=None)
      noisy_data =noisy_data[:-1]
      print(dataname)
      print(len(enh_data))
      pesq_list.append(get_pesq(ori_data,enh_data))
      stoi_list.append(stoi(ori_data, enh_data, fs2, extended=False))
      noisy_pesq_list.append(get_pesq(ori_data,noisy_data))
      noisy_stoi_list.append(stoi(ori_data, noisy_data, fs2, extended=False))
      
      # print(pesq(fs2,ori_data,enh_data))
      # print(stoi(ori_data, enh_data, fs2, extended=False))
  avg_pesq = sum(pesq_list)/len(pesq_list)
  avg_stoi = sum(stoi_list)/len(stoi_list)
  inpainted_trend_avgpesq.append(avg_pesq)
  inpainted_trend_maxpesq.append(max(pesq_list))
  inpainted_trend_minpesq.append(min(pesq_list))
  noisy_avg_pesq = sum(noisy_pesq_list)/len(pesq_list)
  noisy_avg_stoi = sum(noisy_stoi_list)/len(stoi_list)
  noisy_trend_maxpesq.append(max(noisy_pesq_list))
  noisy_trend_minpesq.append(min(noisy_pesq_list))
  noisy_trend_avgpesq.append(noisy_avg_pesq)
  
  print(avg_pesq)
  print(noisy_avg_pesq)
  print(avg_stoi)
  print(noisy_avg_stoi)
  result={
    "inpainted_trend_avgpesq":inpainted_trend_avgpesq,
    "inpainted_trend_maxpesq":inpainted_trend_maxpesq,
    "inpainted_trend_minpesq":inpainted_trend_minpesq,
    "noisy_trend_avgpesq":noisy_trend_avgpesq,
    "noisy_trend_maxpesq":noisy_trend_maxpesq,
    "noisy_trend_minpesq":noisy_trend_minpesq,
    "trend_lossrate":trend_lossrate
  #  "avg_stoi":avg_stoi
  }
  with open ("/E/zhangjinghong/TFGAN-PLC-main/outputs/frnpesq_json/"+str(loss_rate) +"losspesq.json",'w',encoding='UTF-8') as fp:
    fp.write(json.dumps(result,indent=2,ensure_ascii=False))
    print("finish",loss_rate)
# def get_file_info(root_dir):
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith('.wav') or file.endswith('.flac'):
#                 file_path = os.path.join(root, file)

# def cul_SNR(ori_data,enh_data):
#   N = ori_data-enh_data
#   ori_data = ori_data.tolist()
#   N = N.tolist()
#   S = map(lambda x: x**2 , ori_data)
#   S = reduce(lambda x, y: x+y, S)
#   print(S)
#   #N = map(lambda x: x**2 , N)
#   #N = reduce(lambda x, y: x+y, N)

#   from numpy.linalg import norm
#   return 20 * np.log10(norm(ori_data) / norm(N))



# print(len(ori_data),fs1)
# print(len(enh_data),fs2)
# print(pesq(fs2,ori_data,enh_data))
# print(stoi(ori_data, enh_data, fs2, extended=False))
# print(cul_SNR(ori_data,enh_data))
# print(cul_SNR(ori_data,noisy))