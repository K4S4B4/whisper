from torch import Tensor
from torch import nn
import torch

#import cv2

from model import AudioEncoder

class AudioEncoder_Mask(nn.Module):
    def __init__(self, in_audioEncoder: AudioEncoder):
        super().__init__()
        self.audioEncoder = in_audioEncoder

    def forward(self, 
                x: Tensor, #[1,3000,80]
                mask: Tensor,  #[1,3000]
                ):

        x = x * mask

        #xnum = x.squeeze().to('cpu').detach().numpy().copy().astype(np.float32)
        #xnum = xnum.transpose(1,0)
        #cv2.imshow("EncAfterConv", xnum)
        #cv2.waitKey(0)

        xa = self.audioEncoder(x)

        return xa.to(torch.float16)


def export_AudioEncoder_Mask(name, model):
    isMultilingual = not name.endswith('en')
    encoder = AudioEncoder_Mask(model.whisper.encoder)
    device = model.whisper.device

    n_ctx = 1500
    n_mel = n_ctx * 2
    n_ctx_cache = 0

    dummy_mel = torch.randn((1, n_mel, 80), dtype=torch.float32).to(device)
    dummy_mask = torch.cat([torch.ones((1, 1000, 1), dtype=torch.uint8).to(device),
                            torch.zeros((1, 1000, 1), dtype=torch.uint8).to(device),
                            torch.ones((1, 1000, 1), dtype=torch.uint8).to(device)],dim=1)

    inputs = ( dummy_mel,dummy_mask )
    input_names = ['mel','mask']
    output_names = ['audio_feature']

    file_base = "encoder_mask_"
    file_base += str(n_ctx) + "_"
    file_base += str(n_ctx_cache) + "_"
    file_base += name
    file_onnx = file_base + ".onnx"

    torch.onnx.export(encoder,
                    inputs,
                    file_onnx,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=input_names, 
                    output_names=output_names
    )

def model_export(model_name):
    from __init__ import load_model
    model = load_model(model_name, device="cpu")

    export_AudioEncoder_Mask(model_name, model)

if __name__ == '__main__':
    #model_export("tiny")
    #model_export("base")
    #model_export("small")
    model_export("medium")
    #model_export("tiny.en")
    #model_export("base.en")
    #model_export("small.en")
    #model_export("medium.en")
