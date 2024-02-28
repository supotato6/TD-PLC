import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from codec_tcm import Decoder,Encoder
from pesq import pesq
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import wave
from datasets_json_encode import myDataset
#from models.tfgan.generator_json_encode import Generator
from models.tfgan.discriminator_json_encode import Discriminator
from models.stft_loss_json_encode import MultiResolutionSTFTLoss
from optimizsers import RAdam
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
 
	else:
		print("---  There is this folder!  ---")
		

def wav_write(filename, x, fs=16000):

    x = de_emph(x)      # De-emphasis using LPF

    x = x.tolist()       # denormalized
    #x = x.astype('int16')  # cast to int
    #w = wave.Wave_write(filename)
    with wave.open(filename,'wb') as f:
      f.setnchannels(1)
      f.setsampwidth(2)
      f.setframerate(16000)
      f.writeframes(x)
   # w.setparams((1,     # channel
   #              2,     # byte width
   #              fs,    # sampling rate
   #              len(x),  # #. of frames
    #             'NONE',
    #             'not compressed' # no compression
   # ))
   #w.writeframes(array.array('h', x).tobytes())
      #w.close()

    return 0

def get_pesq(ref, deg):
    return pesq(16000, ref, deg, 'nb')

def pingjie(inputs, frame_size, frame_shift):
    nframes = inputs.shape[0]
    sig_len = (nframes - 1) * frame_shift + frame_size
    sig = torch.zeros([sig_len,]).cuda()
    ones = torch.zeros_like(sig).cuda()
    start = 0
    end = start + frame_size
    #print(1)
    for i in range(nframes):

        sig[start:end] += inputs[i, 0 ,:].squeeze(0)
        #sig[start+1280] = 0 

        ones[start:end] += 1
        
        start = start + frame_shift
        end = start + frame_size

    return sig / ones


class TUD(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparam = hparams
        if not os.path.exists(self.hparam.output_path):
            os.makedirs(self.hparam.output_path)
        self.discriminator = Discriminator()
        self.decoder = Decoder()
        self.encoder = Encoder()
        # self.stft_mag = MultiSTFTMag()

    def forward(self, clean):
        """
        Generates a speech using the generator
        given input noise z and noisy speech
        """
        x = self.encoder(clean)
        x = self.decoder(x)
        return x

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for RaLSGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        clean, lens = batch

        # generate speechs
        # z = nn.init.uniform_(torch.Tensor(noisy.shape[0], 100), -1., 1.)
        # z = z.type_as(noisy)
       # print('\n\n\n\n\n\n noisy',noisy.shape)
        g_out = self.encoder(clean)
        g_out = self.decoder(g_out)
        # g_out_mags, clean_mags = self.stft_mag(g_out.squeeze(1), clean.squeeze(1))

        # make discriminator
        disc_real = self.discriminator(clean[:,:,:-36])
        disc_fake = self.discriminator(g_out)

        # Summarize
        tensorboard = self.logger.experiment
        pesq = 3
        #if batch_idx % 1 == 0:
            #tensorboard.add_audio('inpainted speech', pingjie(g_out, self.hparam.slice_len, self.hparam.shift), batch_idx, 16000)
            #tensorboard.add_audio('uninpaint speech', pingjie(noisy, self.hparam.slice_len, self.hparam.shift), batch_idx, 16000)
            #tensorboard.add_audio('clean speech', pingjie(clean, self.hparam.slice_len, self.hparam.shift), batch_idx, 16000)
            #pesq = get_pesq(pingjie(clean[-6000:], self.hparam.slice_len, self.hparam.shift).cpu().detach().numpy(),
                            #pingjie(g_out[-6000:], self.hparam.slice_len, self.hparam.shift).cpu().detach().numpy())
            #tensorboard.add_scalar('pesq', pesq, batch_idx)
            #torchaudio.save(os.path.join('/E/zhangjinghong/TFGAN-PLC-main/outputs/40_loss/noisy', 'test_audio_%d.wav' % batch_idx), pingjie(noisy, self.hparam.slice_len, self.hparam.shift).unsqueeze(0).cpu(), 16000)
            #torchaudio.save(os.path.join('/E/zhangjinghong/TFGAN-PLC-main/outputs/40_loss/clean/', 'test_audio_%d.wav' % batch_idx), pingjie(clean, self.hparam.slice_len, self.hparam.shift).unsqueeze(0).cpu(), 16000)
            #torchaudio.save(os.path.join('/E/zhangjinghong/TFGAN-PLC-main/outputs/40_loss/inpainted/', 'test_audio_%d.wav' % batch_idx), pingjie(g_out, self.hparam.slice_len, self.hparam.shift).unsqueeze(0).cpu(), 16000)
            #print(pingjie(clean, self.hparam.slice_len, self.hparam.shift).cpu().detach().numpy().shape)
            
            #print(batch_idx, 'batch_idx pesq=', pesq)
            #return 0
        #self.log('pesq',pesq)
        # RaLSGAN-GP
        real_logit = disc_real - torch.mean(disc_fake)
        fake_logit = disc_fake - torch.mean(disc_real)

        # Train generator
        if optimizer_idx % 2 < 2:
            #g_loss = (torch.mean((real_logit + 1.) ** 2) + torch.mean((fake_logit - 1.) ** 2)) / 2
            # l1_loss
            l1_loss = 100 * F.mse_loss(g_out, clean[:,:,:-36], reduction='mean')
            # stft_loss
            
            stft_loss = MultiResolutionSTFTLoss()
            g_stft = stft_loss(torch.squeeze(g_out, 1), torch.squeeze(clean[:,:,:-36], 1))
            print(g_stft)
            # g_loss = g_loss + l1_loss + 0.5 * g_stft
            g_loss =  l1_loss + 0.5*g_stft
            g_loss.requires_grad_(True)

            tensorboard.add_scalar('g_loss', g_loss, batch_idx)
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            return g_loss

        # train discriminator
        d_loss = (torch.mean((real_logit - 1.) ** 2) + torch.mean((fake_logit + 1.) ** 2))/2
        d_loss.requires_grad_(True)
        # gradient_penalty = self.compute_gradient_penalty(clean, g_out)
        # energy loss
        # d_loss = d_loss + gradient_penalty

        tensorboard.add_scalar('d_loss', d_loss, batch_idx)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        return d_loss

    def validation_step(self, batch, batch_idx):
        clean, lens = batch

        # z = nn.init.uniform_(torch.Tensor(noisy.shape[0], 100), -1., 1.)
        # z = z.type_as(noisy)
        # generated = self.forward(noisy)
        # l1_loss = 100 * torch.mean(torch.abs(clean - generated))
        # pesq = get_pesq(pingjie(clean, self.hparam.slice_len, self.hparam.shift).cpu().detach().numpy(),
        #                 pingjie(generated, self.hparam.slice_len, self.hparam.shift).cpu().detach().numpy())

        # output = {
        #     'loss': l1_loss,
        #     'pesq': pesq
        # }
        return 0

    def test_step(self, batch, batch_idx):
        clean, lens = batch
        #torch.save(self.encoder,'./encoder.pth')
        # z = nn.init.uniform_(torch.Tensor(noisy.shape[0], 100), -1., 1.)
        # z = z.type_as(noisy)
        test = self.forward(clean)
        clean = clean[:,:,:-36]
        save_path_clean = os.path.join(self.hparam.output_path, 'clean/test_audio_%d.wav' % batch_idx)
        save_path_test = os.path.join(self.hparam.output_path, 'encoded/test_audio_%d.wav' % batch_idx)
        #save_path_noisy = os.path.join(self.hparam.output_path, 'noisy/t  est_audio_%d.wav' % batch_idx)
        print(pingjie(clean, self.hparam.slice_len, self.hparam.shift).cpu())
        #wav_write(save_path_clean, pingjie(clean, self.hparam.slice_len, self.hparam.shift).cpu(), 16000)
        torchaudio.save(save_path_clean, pingjie(clean, self.hparam.slice_len, self.hparam.shift).unsqueeze(0).cpu(), 16000)
        torchaudio.save(save_path_test, pingjie(test, self.hparam.slice_len, self.hparam.shift).unsqueeze(0).cpu(), 16000)
        #torchaudio.save(save_path_test, test, 16000)
        print('Successfully inpainted %d audios' % (batch_idx+1))

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # optimizer_g = RAdam(self.generator.parameters(), lr=0.0002, betas=(0.9, 0.999))
        # optimizer_d = RAdam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))
        optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=0.00040, betas=(0.7, 0.99))
        optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.00040, betas=(0.7, 0.99))
        #optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))
        # optimizer_stftd = RAdam(self.dc_discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))

        return [optimizer_decoder,optimizer_encoder], []

    def collate_fn(self, batch):
        # print(batch[0][0].size())
        # for item in batch:
            # print(item[0].size())
        # print('noisy size = ', noisy.size())
        clean = torch.cat([item[0].unsqueeze(1) for item in batch], 0)
        lens = torch.LongTensor([item[1] for item in batch])
        return clean, lens

    def train_dataloader(self):
        # REQUIRED
        print(1111111111111111111111111111111111111111111111111111111111111111111111)
        print("training!!!!",self.hparam.data_dir)
        return DataLoader(
            myDataset(self.hparam.data_dir, self.hparam.slice_len, loss_rate=self.hparam.loss_rate),
            batch_size=self.hparam.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        print(1111111111111111111111111111111111111111111111111111111111111111111111)
        return DataLoader(
            myDataset(self.hparam.val_data_dir, self.hparam.slice_len, loss_rate=10),
            batch_size=1, collate_fn=self.collate_fn, shuffle=False, num_workers=4)

    def test_dataloader(self):
        print(1111111111111111111111111111111111111111111111111111111111111111111111)
        return DataLoader(
            myDataset(self.hparam.test_data_dir, self.hparam.slice_len, loss_rate=self.hparam.loss_rate),
            batch_size=1, collate_fn=self.collate_fn, shuffle=False, num_workers=4)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--shift', default=160, type=int)
    parser.add_argument('--slice_len', default=480, type=int)
    # path/E/zhangjinghong/TFGAN-PLC-main/data_m/trend_data/
    parser.add_argument('--data_dir', default='/E/zhangjinghong/TFGAN-PLC-main/data_m/', type=str)
    parser.add_argument('--val_data_dir', default='/', type=str)
    parser.add_argument('--output_path', default='/E/zhangjinghong/wav2v/outputs/40_rongyu', type=str)
    parser.add_argument('--test_data_dir', default='/E/zhangjinghong/TFGAN-PLC-main/data_m/', type=str)
    parser.add_argument('--max_epochs', default=120, type=int)
    # parse params
    hparams = parser.parse_args()
    print(hparams.data_dir)
    hparams.loss_rate = 1.0
    model = TUD(hparams)

    checkpoint_callback = ModelCheckpoint(dirpath='/E/zhangjinghong/wav2v/checkpoint/checkpoints.ckpt', save_top_k=3, verbose=True,
                                          monitor='pesq', mode='max')

    tb_logger = pl_loggers.TensorBoardLogger('./logs/')
    print('Training has started. Please use \'tensorboard --logdir=./logs\' to monitor.')

    trainer = pl.Trainer(gpus='1', callbacks=[checkpoint_callback],enable_checkpointing=True,
                        resume_from_checkpoint='/E/zhangjinghong/wav2v/checkpoint/checkpoint'+'100'+'.ckpt',
                        logger=tb_logger,max_epochs = hparams.max_epochs)
    print(hparams.mode)
    print()
    if hparams.mode == 'train':
        trainer.fit(model)
        trainer.save_checkpoint("./checkpoint/checkpoint"+str(hparams.max_epochs)+".ckpt")
    elif hparams.mode == 'test':
        
        hparams.loss_rate = 1.0
        model = TUD(hparams)
        #ckpt = torch.load('/E/zhangjinghong/wav2v/checkpoint/checkpoint'+'15'+'.ckpt')
        #model.load_state_dict(ckpt['model_state_dict'])
        #model1 = model.decoder()
        #print(model1.state_dict())
        model.freeze()
        trainer.test(model)
        # for i in range(10,100):
        #     hparams.loss_rate = i/10
        #     hparams.output_path = '/E/zhangjinghong/TFGAN-PLC-main/outputs/'+ str(hparams.loss_rate)+'_loss'
        #     mkdir(os.path.join(hparams.output_path,'clean'))
        #     mkdir(os.path.join(hparams.output_path,'noisy'))
        #     mkdir(os.path.join(hparams.output_path,'inpainted'))
        #     model = TFGAN(hparams)
        #     model.freeze()
        #     trainer.test(model)
