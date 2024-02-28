import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import fairseq

import os
import glob
import json
import random
from random import randint

class myDataset(Dataset):
    def __init__(self, data_dir, slice_len, loss_rate=20):
        self.data_dir = data_dir
        self.model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/E/zhangjinghong/wav2v/vq_pretrain_model/vq/vq-wav2vec.pt'])
        self.model = self.model[0]
        self.loss_rate = loss_rate
        print(self.data_dir)
        
        #self.clean_root =os.path.join(self.data_dir, 'train/clean_trainset_wav/')
        self.clean_root =os.path.join(self.data_dir, 'test/clean_testset_wav/')
        print('Training data set is', self.clean_root)
        #self.clean_root = "/E/zhangjinghong/TFGAN-PLC-main/data_m/train/clean_trainset_wav/"
        #self.clean_root = "/media/ailab/E/zhangjinghong/wav2vec/data/clean_trainset_wav/"
        self.clean_path = self.get_path(self.clean_root)

        self.slice_len = slice_len

    def get_path(self, root):
        paths = glob.glob(os.path.join(root, '*.wav'))
        return sorted(paths)

    def get_lossjson_path(self, root):
        paths = glob.glob(os.path.join(root, '*.json'))
        return sorted(paths)
    
    def get_jsondata(self, root):
        with open(root) as f:
            jsondata = json.load(f)
        return jsondata
          
    # def padding(self, x):
    #     x.unsqueeze_(0)
    #     len_x = x.size(-1)
    #     pad_len = self.stride - len_x % self.stride
    #     return F.pad(x, (pad_len, 0), zmode='constant')

    def truncate(self, n, c):
        offset = 960
        length = n.size(-1)
        start = torch.randint(length - offset, (1,))
        return n[:, start:start + offset], c[:, start:start + offset]

    def signal_to_frame(self, c, frame_size=480, frame_shift=160, ctex_len = [800,2400]):
        sig_len = len(c)
        if sig_len>48000:
            st = randint(0,sig_len-48000)
            c = c[st:st+48000]
            sig_len = len(c)
        nframes = (sig_len // frame_shift)
        c_slice = np.zeros([nframes, frame_size+36])
        start = 0
        end = start + frame_size
        k = 0
        self.model.eval()
        for i in range(nframes):
            if start <ctex_len[0]:
                ctex_slice = torch.cat([torch.zeros(ctex_len[0]-start),torch.from_numpy(c[:start+ctex_len[1]])])
            elif sig_len < start+ctex_len[1]:
                ctex_slice = torch.cat([torch.from_numpy(c[start-ctex_len[0]:]),torch.zeros(start+ctex_len[1]-sig_len)])
                
            else:
                ctex_slice =torch.from_numpy(c[start-ctex_len[0]:start+ctex_len[1]])
            ctex_slice = ctex_slice.float()
            ctex_slice = ctex_slice.reshape(1,3200)
            #print(ctex_slice.shape,"djhf3ioquthieut3ti",ctex_slice)
            z = self.model.feature_extractor(ctex_slice)
            # print(z.shape)
            _, idxs = self.model.vector_quantizer.forward_idx(z)
            idxs = idxs.reshape(1,36)
            if end < sig_len:
                c_slice[i, :-36] = c[start:end]
                k += 1
            else:
                tail_size = sig_len - start
                c_slice[i, :tail_size] = c[start:]
            c_slice[i, -36:] = idxs
            start = start + frame_shift
            end = start + frame_size
        return c_slice

    def __len__(self):
        return len(self.clean_path)
    def get_json(self, path):
        with open(path) as f :
            data = json.load(f)
        return data
    def __getitem__(self, idx):
        #print(self.clean_path)
        clean = sf.read(self.clean_path[idx])[0]
        #print(self.json_path)
        #print(clean.shape,noisy.shape)
        clean_slice= self.signal_to_frame(clean)
        #print()
        length = clean_slice.shape[1]
        clean = torch.FloatTensor(clean_slice)

        #print(noisy.shape)
        #print('abaaba',noisy.shape)
        # clean = self.padding(clean).squeeze(0)
        # noisy = self.padding(noisy).squeeze(0)

        # clean /= clean.abs().max()
        # noisy /= noisy.abs().max()

        return clean,length
def collate_fn(self, batch):
    # print(batch[0][0].size())
    for item in batch:
        # print(item[0].size())
    # print('noisy size = ', noisy.size())
        clean = torch.cat([item[1].unsqueeze(1) for item in batch], 0)
        lens = torch.LongTensor([item[2] for item in batch])
        return clean, lens

def train_dataloader(hparam,collate_fn):
    # REQUIRED
    return DataLoader(
        myDataset(hparam.data_dir, hparam.slice_len, loss_rate=hparam.loss_rate),
        batch_size=hparam.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--shift', default=960, type=int)
    parser.add_argument('--slice_len', default=1920, type=int)
    # path/E/zhangjinghong/TFGAN-PLC-main/data_m/trend_data/
    parser.add_argument('--data_dir', default='/media/ailab/E/zhangjinghong/TFGAN/data_m/', type=str)
    parser.add_argument('--val_data_dir', default='/', type=str)
    parser.add_argument('--output_path', default='/media/ailab/E/zhangjinghong/TFGAN/outputs/40_rongyu', type=str)
    parser.add_argument('--test_data_dir', default='/media/ailab/E/zhangjinghong/TFGAN/data_m/', type=str)
    parser.add_argument('--resume', default=None, type=str)
    # parse params
    hparams = parser.parse_args()
    a = train_dataloader(hparam,collate_fn)


