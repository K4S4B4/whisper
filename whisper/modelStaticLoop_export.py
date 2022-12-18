import torch
#import onnx

from modelStaticLoop import TextDecoder_StaticLoop, TextDecoder_ForcedAlignment

def export_TextDecoder_StaticLoop(name, model, n_batch: int, n_ctx_in: int, n_ctx_out: int, n_audioFeature: int, makeOnnxAttentionPastPresent: bool, ioInt: int, doUseOffset: bool):
    isMultilingual = not name.endswith('en')
    device = model.whisper.device
    n_state = model.whisper.dims.n_text_state
    n_layer = model.whisper.dims.n_text_layer
    if ioInt == 32:
        intDtype = torch.int32
    else:
        intDtype = torch.int64

    if n_ctx_out == 0:
        decoder = TextDecoder_ForcedAlignment(model.whisper.decoder, n_ctx_out, isMultilingual, makeOnnxAttentionPastPresent)
        file_base = "decoder_fa_a16_"
        output_names = ['out_probs']
    else:
        decoder = TextDecoder_StaticLoop(model.whisper.decoder, n_ctx_out, isMultilingual, makeOnnxAttentionPastPresent, ioInt)
        file_base = "decoder_sl_a16_"
        output_names = ['out_tokens']

    input_names = ['in_tokens', 'audio_feature']
    if doUseOffset:
        file_base += "pos_"
        input_names.append('offset')

    file_base += str(n_batch) + "_"
    file_base += str(n_ctx_in) + "_"
    file_base += str(n_ctx_out) + "_"
    file_base += str(n_audioFeature) + "_"
    file_base += "int" + str(ioInt) + "_"
    file_base += name
    file_onnx = file_base + ".onnx"

    dynamic_axes = dict()
    if n_batch < 0:
        dynamic_axes['in_tokens'] = {0: 'n_batch'}
        dynamic_axes['audio_feature'] = {0: 'n_batch'}
        n_batch = 1
    if n_ctx_in < 0:
        if 'in_tokens' in dynamic_axes.keys():
            dynamic_axes['in_tokens'][1] = 'n_tokens_in'
        else:
            dynamic_axes['in_tokens'] = {1: 'n_tokens_in'}
        n_ctx_in = 4
    if n_audioFeature < 0:
        if 'audio_feature' in dynamic_axes.keys():
            dynamic_axes['audio_feature'][1] = 'n_audio_feature'
        else:
            dynamic_axes['audio_feature'] = {1: 'n_audio_feature'}
        n_audioFeature = 1500

    token_list = []
    for i in range(n_ctx_in):
        token_list.append(torch.tensor(i, dtype=intDtype).to(device))
    dummy_tokens = torch.stack(token_list).unsqueeze(0)
    dummy_tokens = dummy_tokens.expand(n_batch, n_ctx_in)
    dummy_audioFeature = torch.randn((1, n_audioFeature, n_state), dtype=torch.float16).to(device)

    inputs = [ dummy_tokens, dummy_audioFeature ]
    if doUseOffset:
        dummy_offset = torch.ones((1,1), dtype=intDtype).to(device) * n_ctx_in
        inputs.append(dummy_offset)

    torch.onnx.export(decoder,
                    tuple(inputs),
                    file_onnx,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=input_names, 
                    output_names=output_names,
                    dynamic_axes=dynamic_axes
    )

def simplify_TextDecoder_StaticLoop(name, n_ctx_in: int, n_ctx_out: int):
    from onnxsim import simplify

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

    #export_TextDecoder_StaticLoop(model_name, model, 8, 8, True)
    #export_TextDecoder_StaticLoop(model_name, model, 16, 16, -1, True)
    #export_TextDecoder_StaticLoop(model_name, model, 32, 32, True)

    #export_TextDecoder_StaticLoop(model_name, model, -1, 3, -1, True)
    #export_TextDecoder_StaticLoop(model_name, model, 8, -1, 1, 1500, True)
    #export_TextDecoder_StaticLoop(model_name, model, 1, 32, 1, 1500, True, 32)
    #export_TextDecoder_StaticLoop(model_name, model, 1, -1, 1, 1500, True, 32)

    #export_TextDecoder_StaticLoop(model_name, model, 8, 2, True)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 4)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 16)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 32)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 64)
    #export_TextDecoder_StaticLoop(model_name, model, 32, 64)

    # forced alignment v1.0
    export_TextDecoder_StaticLoop(model_name, model, 8, -1, 0, 1500, True, 32, False)

    ## static loop v1.0
    #export_TextDecoder_StaticLoop(model_name, model, 1, -1, 3, 1500, True, 32)
    #export_TextDecoder_StaticLoop(model_name, model, 1, -1, 8, 1500, True, 32, True)
    #export_TextDecoder_StaticLoop(model_name, model, 1, -1, 16, 1500, True, 32, True)

if __name__ == '__main__':
    #model_name = "tiny"
    #model_name = "base"
    #model_name = "small"
    #model_name = "medium"
    #model_name = "tiny.en"
    #model_name = "base.en"
    #model_name = "small.en"
    #model_name = "medium.en"

    executeExport("tiny")
    executeExport("base")
    #executeExport("small")
    executeExport("medium")
    executeExport("tiny.en")
    executeExport("base.en")
    executeExport("small.en")
    executeExport("medium.en")

    #executeSimplify(model_name)