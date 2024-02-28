import torch
import fairseq

#cp = torch.load('/media/ailab/E/zhangjinghong/wav2vec/model/vq/vq-wav2vec.pt')
#print(cp.filenames)
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/media/ailab/E/zhangjinghong/wav2vec/vq_pretain_model/vq/vq-wav2vec.pt'])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,3200)
print(wav_input_16khz.dtype)
z = model.feature_extractor(wav_input_16khz)
print(z.shape)
_, idxs = model.vector_quantizer.forward_idx(z)
print(idxs.shape) # o

