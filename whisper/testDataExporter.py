from audio import log_mel_spectrogram
import torch
import numpy as np
from struct import unpack, pack

def gen_mel():
    audio_path = "tests/jfk.flac"
    mel = log_mel_spectrogram(audio_path).unsqueeze(0)
    mel_t = mel.permute(0, 2, 1)

    n_pad = 3000 - mel_t.shape[1]
    pad = torch.zeros(1,n_pad,80)
    mel_t = torch.cat([mel_t, pad], dim=1)
    return mel_t

def exportTensor(file_name, tensor):
    data = tensor.reshape([-1]).to('cpu').detach().numpy().copy().astype(np.float32)

    f = open(file_name, 'wb')
    for x in data:
        b = pack('f', x)
        f.write(b)
    f.close()

mel_t_pad = gen_mel()



