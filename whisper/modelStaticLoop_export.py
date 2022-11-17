import torch

import onnx
from onnxsim import simplify

from modelStaticLoop import TextDecoder_StaticLoop

def export_TextDecoder_StaticLoop(name, model, n_ctx_in: int, n_ctx_out: int, makeOnnxAttentionPastPresent: bool):
    isMultilingual = not name.endswith('en')
    decoder = TextDecoder_StaticLoop(model.whisper.decoder, n_ctx_out, isMultilingual, makeOnnxAttentionPastPresent)
    device = model.whisper.device
    n_state = model.whisper.dims.n_text_state
    n_layer = model.whisper.dims.n_text_layer

    token_list = []
    for i in range(n_ctx_in):
        token_list.append(torch.tensor(i, dtype=torch.int64).to(device))
    dummy_tokens = torch.stack(token_list).unsqueeze(0)
    dummy_audioFeature = torch.randn((1, 1500, n_state), dtype=torch.float32).to(device)

    inputs = ( dummy_tokens, dummy_audioFeature )
    input_names = ['in_tokens', 'audio_feature']
    output_names = ['out_tokens']

    file_base = "decoder_staticLoop_"
    file_base += str(n_ctx_in) + "_"
    file_base += str(n_ctx_out) + "_"
    file_base += name
    file_onnx = file_base + ".onnx"
    file_simp = file_base + "_smpl.onnx"

    torch.onnx.export(decoder,
                    inputs,
                    file_onnx,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=input_names, 
                    output_names=output_names
    )
    #onnx_model = onnx.load(f'{file_onnx}')
    #onnx_model_simp, check = simplify(onnx_model)
    #onnx.save(onnx_model_simp, f'{file_simp}')


def simplify_TextDecoder_StaticLoop(name, n_ctx_in: int, n_ctx_out: int):
    file_base = "decoder_staticLoop_"
    file_base += str(n_ctx_in) + "_"
    file_base += str(n_ctx_out) + "_"
    file_base += name
    file_onnx = file_base + ".onnx"
    file_simp = file_base + "_smpl.onnx"

    onnx_model = onnx.load(f'{file_onnx}')
    onnx_model_simp, check = simplify(onnx_model)
    onnx.save(onnx_model_simp, f'{file_simp}')

def executeSimplify(model_name):
    #simplify_TextDecoder_StaticLoop(model_name, 8, 2)
    #simplify_TextDecoder_StaticLoop(model_name, 8, 4)
    #simplify_TextDecoder_StaticLoop(model_name, 8, 8)
    #simplify_TextDecoder_StaticLoop(model_name, 8, 16)
    #simplify_TextDecoder_StaticLoop(model_name, 8, 32)

    #simplify_TextDecoder_StaticLoop(model_name, 16, 2)
    #simplify_TextDecoder_StaticLoop(model_name, 16, 3)
    simplify_TextDecoder_StaticLoop(model_name, 9, 3)


def executeExport(model_name):
    from __init__ import load_model
    model = load_model(model_name, device="cpu")

    # cacheReturnRule = 0 : return appended self cache
    # cacheReturnRule = 3 : return appended self cache for Onnx Attention node

    export_TextDecoder_StaticLoop(model_name, model, 8, 8, True)
    export_TextDecoder_StaticLoop(model_name, model, 16, 16, True)
    export_TextDecoder_StaticLoop(model_name, model, 32, 32, True)
    #export_TextDecoder_StaticLoop(model_name, model, 9, 3, False)

    #export_TextDecoder_StaticLoop(model_name, model, 8, 4)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 16)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 32)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 64)
    #export_TextDecoder_StaticLoop(model_name, model, 32, 64)

if __name__ == '__main__':
    #model_name = "tiny"
    #model_name = "base"
    model_name = "small"
    #model_name = "medium"
    #model_name = "tiny.en"
    #model_name = "base.en"
    #model_name = "small.en"
    #model_name = "medium.en"

    executeExport(model_name)
    #executeSimplify(model_name)