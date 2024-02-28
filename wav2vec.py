# -*- coding: utf-8 -*-
# coding:unicode_escape
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,AutoProcessor

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


#librispeech_samples_ds = load_dataset("/media/ailab/E/zhangjinghong/TFGAN/data_m/test/test/", "clean", split="test")

# load audio
audio_input, sample_rate = (torch.randn(10000),16000)

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# INFERENCE

# retrieve logits & take argmax
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
print(logits.shape)
# transcribe
transcription = processor.decode(predicted_ids[0])
#print(transcription)
# # FINE-TUNE

# target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

# # encode labels
# with processor.as_target_processor():
#   labels = processor(target_transcription, return_tensors="pt").input_ids

# # compute loss by passing labels
# loss = model(input_values, labels=labels).loss
# loss.backward()