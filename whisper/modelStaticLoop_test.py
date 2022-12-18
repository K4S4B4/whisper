import os.path
import numpy as np
import torch
import onnxruntime
import time
from struct import unpack, pack

import cv2

from audio import load_audio, log_mel_spectrogram, SAMPLE_RATE
from modelStaticLoop import TextDecoder_StaticLoop, TextDecoder_ForcedAlignment, TextDecoder_dynamicLoop
from tokenizer import get_tokenizer

def gen_audio_feature_zeros(model):
    return torch.zeros((1,1500,model.whisper.dims.n_text_state), dtype=torch.float32).to(model.whisper.device)

def gen_mel_zeros(model):
    return torch.zeros((1,3000,80), dtype=torch.float32).to(model.whisper.device)

def gen_mel(model):
    audio_path = "tests/jfk.flac"
    mel = log_mel_spectrogram(audio_path).unsqueeze(0)
    mel_t = mel.permute(0, 2, 1)

    n_pad = 3000 - mel_t.shape[1]
    pad = torch.zeros(1,n_pad,80)
    mel_t = torch.cat([mel_t, pad], dim=1).to(model.whisper.device)
    return mel_t

def load_mel(device):
    data_path = "tests/mel_t_pad.dat"
    data = []
    with open(data_path, "rb") as f:
        while True:
            b = f.read(4)
            if len(b) == 0:
                break
            x = unpack('f', b)
            data.append(x)

    tensor = torch.tensor(data).to(torch.float32).to(device)
    tensor = tensor.reshape([1,3000,80])
    return tensor

def load_audio_feature(size, n_state, device):
    data_path = "tests/audio_feature_" + size + ".dat"
    data = []
    with open(data_path, "rb") as f:
        while True:
            b = f.read(4)
            if len(b) == 0:
                break
            x = unpack('f', b)
            data.append(x)

    tensor = torch.tensor(data).to(torch.float32).to(device)
    tensor = tensor.reshape([1,1500,n_state])
    return tensor

def gen_audio_feature(model):
    #mel_t = gen_mel(model)
    mel_t = load_mel(model.whisper.device)
    encoder = model.whisper.encoder
    return encoder(mel_t)

def gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in: int):
    return torch.zeros((1, n_ctx_in), dtype=torch.int64)

def gen_tokens(isMultilingual, tokenizer, n_ctx_in: int):
    in_tokens = torch.ones((1, n_ctx_in), dtype=torch.int64)
    if isMultilingual:
        in_tokens[:,n_ctx_in-1] = tokenizer.timestamp_begin
        in_tokens[:,n_ctx_in-2] = 50359
        in_tokens[:,n_ctx_in-3] = 50259
        in_tokens[:,n_ctx_in-4] = tokenizer.sot
        in_tokens[:,n_ctx_in-5] = tokenizer.timestamp_begin + 1499
        in_tokens[:,n_ctx_in-6] = tokenizer.timestamp_begin + 1499
        in_tokens[:,0] = tokenizer.sot_prev
    else:
        in_tokens[:,n_ctx_in-1] = tokenizer.timestamp_begin
        in_tokens[:,n_ctx_in-2] = tokenizer.sot
        in_tokens[:,n_ctx_in-3] = tokenizer.timestamp_begin + 1499
        in_tokens[:,n_ctx_in-4] = tokenizer.timestamp_begin + 1499
        in_tokens[:,0] = tokenizer.sot_prev
    return in_tokens

def gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in: int, words: str):
    tokens_toAlign = tokenizer.encode(words)
    
    in_tokens = torch.ones((1, n_ctx_in), dtype=torch.int64)
    if isMultilingual:
        in_tokens[:,0] = tokenizer.sot
        in_tokens[:,1] = 50259
        in_tokens[:,2] = 50359
        in_tokens[:,3] = tokenizer.timestamp_begin

        for i in range(n_ctx_in - 4):
            if len(tokens_toAlign) > i:
                in_tokens[:,i + 4] = tokens_toAlign[i]
            else:
                in_tokens[:,i + 4] = tokenizer.eot

    else:
        in_tokens[:,0] = tokenizer.sot
        in_tokens[:,1] = tokenizer.timestamp_begin

        for i in range(n_ctx_in - 2):
            if len(tokens_toAlign) > i:
                in_tokens[:,i + 2] = tokens_toAlign[i]
            else:
                in_tokens[:,i + 2] = tokenizer.eot

    return in_tokens

def testTorch_AudioEncoder(name, model, n_ctx_in: int, n_ctx_out: int):
    mel_t_zeros = gen_mel_zeros(model)
    mel_t = gen_mel(model)

    encoder = model.whisper.encoder

    # warm up
    for k in range(5):
        out_audio_feature = encoder(mel_t_zeros)

    inference_start = time.time()
    out_audio_feature = encoder(mel_t)
    #print(out_audio_feature[0,0,0])
    print("PyTorch Inference took:", (time.time() - inference_start) * 1000, "ms")

def testOnnx_AudioEncoder(name, model, n_ctx_in: int, n_ctx_out: int, gen:str):
    mel_t_zeros = gen_mel_zeros(model)
    mel_t = gen_mel(model)

    ###################################################################
    sess_options = onnxruntime.SessionOptions()
    #providers = ['DmlExecutionProvider']
    #providers = ['CUDAExecutionProvider']
    providers = ['CPUExecutionProvider']
    #sess_options.log_severity_level = 0
    #sess_options.log_verbosity_level = 1
    #sess_options.enable_profiling = True
    
    if gen == "raw":
        model_path = f'encoder_mask_{n_ctx_in}_{n_ctx_out}_{name}.onnx'
    elif gen == "opt":
        model_path = f'encoder_mask_{n_ctx_in}_{n_ctx_out}_{name}_opt.onnx'
    elif gen == "opt_fp16":
        model_path = f'encoder_mask_{n_ctx_in}_{n_ctx_out}_{name}_opt_fp16.onnx'
    #model_path = f'encoder_org_{n_ctx_in}_{n_ctx_out}_{name}.onnx'
    #model_path = f'encoder_org_{n_ctx_in}_{n_ctx_out}_{name}_opt.onnx'

    load_start = time.time()
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    print("Load took:", time.time() - load_start, "s")

    # warm up
    ort_inputs = {
        'mel':  mel_t_zeros.to('cpu').detach().numpy().copy().astype(np.float32),
        'mask':  np.ones((1,3000,1), dtype=np.uint8)
    }
    for k in range(5):
        out_audio_feature = session.run(None, ort_inputs)

    ort_inputs = {
        'mel':  mel_t.to('cpu').detach().numpy().copy().astype(np.float32),
        'mask':  np.ones((1,3000,1), dtype=np.uint8)
    }
    inference_start = time.time()
    audio_feature = session.run(None, ort_inputs)
    #print(out_audio_feature[0][0,0,0])
    print("ONNX RT Inference took:", (time.time() - inference_start) * 1000, "ms")
    ###################################################################

    #xa = audio_feature[0].squeeze(0).transpose()
    #cv2.imshow("Enc_xa", xa)
    #cv2.waitKey(1)

    ## decode
    ####################################################################
    #isMultilingual = not name.endswith('en')
    #tokenizer = get_tokenizer(multilingual=isMultilingual)
    #in_tokens = gen_tokens(isMultilingual, tokenizer, 8).to(model.whisper.device)
    #decoder = TextDecoder_StaticLoop(model.whisper.decoder, 32, isMultilingual)

    #audio_feature = torch.from_numpy(audio_feature[0].astype(np.float32)).clone().to(model.whisper.device)
    #out_tokens = decoder(in_tokens, audio_feature)

    #out_token_list = []
    #for i in range(32):
    #    out_token_list.append(out_tokens[0, i])
    #text = tokenizer.decode(out_token_list)

    #print("PyTorch:", text)
    ####################################################################

def testTorch_TextDecoder_StaticLoop(name, model, n_ctx_in: int, n_ctx_out: int, makeOnnxAttentionPastPresent: bool):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    audio_feature = gen_audio_feature(model).to(model.whisper.device)
    in_tokens = gen_tokens(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)
    audio_feature_zeros = gen_audio_feature_zeros(model)
    in_tokens_zeros = gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)

    decoder = TextDecoder_StaticLoop(model.whisper.decoder, n_ctx_out, isMultilingual, makeOnnxAttentionPastPresent, 64)

    # warm up
    #for k in range(5):
    #    out_tokens = decoder(in_tokens_zeros, audio_feature_zeros)

    inference_start = time.time()
    out_tokens = decoder(in_tokens, audio_feature)
    print("PyTorch Inference took:", (time.time() - inference_start) * 1000, "ms")

    out_token_list = []
    for i in range(n_ctx_out):
        out_token_list.append(out_tokens[0, i])
    text = tokenizer.decode(out_token_list)

    print("PyTorch:", text)

def testTorch_TextDecoder_DynamicLoop(name, model, n_ctx_in: int, n_ctx_out: int):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    audio_feature = gen_audio_feature(model).to(model.whisper.device)
    in_tokens = gen_tokens(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)
    audio_feature_zeros = gen_audio_feature_zeros(model)
    in_tokens_zeros = gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)
    in_positioins = torch.arange(0, in_tokens.shape[1]).unsqueeze(0).to(model.whisper.device)

    decoder = TextDecoder_dynamicLoop(model.whisper.decoder, n_ctx_out, isMultilingual, model.whisper.dims.n_text_layer)

    # warm up
    for k in range(3):
        out_tokens = decoder.run(in_tokens_zeros, in_positioins, audio_feature_zeros)

    inference_start = time.time()
    out_tokens = decoder.run(in_tokens, in_positioins, audio_feature)
    print("PyTorch Inference took:", (time.time() - inference_start) * 1000, "ms")

    #out_token_list = []
    #for i in range(out_tokens.shape[1]):
    #    out_token_list.append(out_tokens[0, i])
    #text = tokenizer.decode(out_token_list)
    text = tokenizer.decode(out_tokens[0])

    print("PyTorch:", text)

def testOnnx_TextDecoder_DynamicLoop(name, model, n_ctx_in: int):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    audio_feature = gen_audio_feature(model).to(model.whisper.device)
    in_tokens = gen_tokens(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)
    audio_feature_zeros = gen_audio_feature_zeros(model)
    in_tokens_zeros = gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)
    in_positioins = torch.arange(0, in_tokens.shape[1]).unsqueeze(0).to(model.whisper.device)

    ###################################################################
    sess_options = onnxruntime.SessionOptions()
    providers = ['CUDAExecutionProvider']
    #providers = ['CPUExecutionProvider']
    #sess_options.log_severity_level = 0
    #sess_options.log_verbosity_level = 1
    #sess_options.enable_profiling = True
    
    model_path = f'decoder_dl_a16_-1_128_1500_{name}_opt_fp16.onnx'

    load_start = time.time()
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    print("Load took:", time.time() - load_start, "s")

    # warm up
    ort_inputs = {
        'in_tokens':  in_tokens_zeros.to('cpu').detach().numpy().copy().astype(np.int64),
        'in_positions':  in_positioins.to('cpu').detach().numpy().copy().astype(np.int32),
        'audio_feature':  audio_feature_zeros.to('cpu').detach().numpy().copy().astype(np.float16)
    }
    for k in range(3):
        ort_outputs = session.run(None, ort_inputs)

    ort_inputs = {
        'in_tokens':  in_tokens.to('cpu').detach().numpy().copy().astype(np.int64),
        'in_positions':  in_positioins.to('cpu').detach().numpy().copy().astype(np.int32),
        'audio_feature':  audio_feature.to('cpu').detach().numpy().copy().astype(np.float16)
    }
    inference_start = time.time()
    ort_outputs = session.run(None, ort_inputs)
    duration = time.time() - inference_start
    print("ONNX RT Inference took:", duration * 1000, "ms")
    print("ave one inferc Dynamic:", duration * 1000 / (len(ort_outputs[0][0]) - 1), "ms")
    ###################################################################
    text = tokenizer.decode(ort_outputs[0][0])

    print("PyTorch:", text)

def testTorch_TextDecoder_ForcedAlignment(name, model, n_ctx_in: int, n_ctx_out: int, makeOnnxAttentionPastPresent: bool):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    audio_feature = gen_audio_feature(model).to(model.whisper.device)
    audio_feature_zeros = gen_audio_feature_zeros(model)
    in_tokens_zeros = gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)

    in_tokens0 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.").to(model.whisper.device)
    in_tokens1 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " and so my fellow Americans. Ask not what you want for free for you. Ask what you want to do for the community.").to(model.whisper.device)
    in_tokens2 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " shloud I speak now? ah, and so my fellow Americans ask not what your country can do for you ask what you can do for your country").to(model.whisper.device)
    in_tokens3 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " ah year oh my god ask not what your country can do for you ask what you can do for your country").to(model.whisper.device)
    in_tokens = torch.cat([in_tokens0,in_tokens1,in_tokens2,in_tokens3], dim=0)

    decoder = TextDecoder_ForcedAlignment(model.whisper.decoder, n_ctx_out, isMultilingual, makeOnnxAttentionPastPresent)

    # warm up
    for k in range(3):
        out_alignProbs = decoder(in_tokens_zeros, audio_feature_zeros)

    inference_start = time.time()
    out_alignProbs = decoder(in_tokens, audio_feature)
    print("PyTorch Inference took:", (time.time() - inference_start) * 1000, "ms")

    for b in range(in_tokens.shape[0]):
        for i in range(n_ctx_in - 1):
            print(in_tokens[b, i + 1].to('cpu').detach().numpy().copy(), tokenizer.decode(in_tokens[b, i + 1]), out_alignProbs[b, i].to('cpu').detach().numpy().copy())
        print("")

def testOnnx_TextDecoder_ForcedAlignment(name, model, n_ctx_in: int, isDynamic: bool):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    audio_feature = gen_audio_feature(model).to(model.whisper.device)
    audio_feature_zeros = gen_audio_feature_zeros(model)
    in_tokens_zeros = gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in).to(model.whisper.device)

    in_tokens0 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.").to(model.whisper.device)
    in_tokens1 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " and so my fellow Americans. Ask not what you want for free for you. Ask what you want to do for the community.").to(model.whisper.device)
    in_tokens2 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " shloud I speak now? ah, and so my fellow Americans ask not what your country can do for you ask what you can do for your country").to(model.whisper.device)
    in_tokens3 = gen_tokens_toAlign(isMultilingual, tokenizer, n_ctx_in, " ah year oh my god ask not what your country can do for you ask what you can do for your country").to(model.whisper.device)
    in_tokens = torch.cat([in_tokens0,in_tokens1,in_tokens2,in_tokens3], dim=0)

    ###################################################################
    sess_options = onnxruntime.SessionOptions()
    providers = ['CUDAExecutionProvider']
    
    if isDynamic:
        model_path = f'decoder_fa_a16_8_-1_0_1500_int32_{name}_opt_fp16.onnx'
    else:
        model_path = f'decoder_fa_a16_4_32_0_1500_int32_{name}_opt_fp16.onnx'

    load_start = time.time()
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    print("Load took:", time.time() - load_start, "s")

    in_tokens_zeros_4 = torch.cat([in_tokens_zeros, in_tokens_zeros, in_tokens_zeros, in_tokens_zeros], dim=0)
    # warm up
    ort_inputs = {
        'in_tokens':  in_tokens_zeros_4.to('cpu').detach().numpy().copy().astype(np.int32),
        'audio_feature': audio_feature_zeros.to('cpu').detach().numpy().copy().astype(np.float16)
    }
    for k in range(3):
        out_alignProbs = session.run(None, ort_inputs)[0]

    ort_inputs = {
        'in_tokens':  in_tokens.to('cpu').detach().numpy().copy().astype(np.int32),
        'audio_feature': audio_feature.to('cpu').detach().numpy().copy().astype(np.float16)
    }
    inference_start = time.time()
    out_alignProbs = session.run(None, ort_inputs)[0]
    print("ONNX RT Inference took:", (time.time() - inference_start) * 1000, "ms")
    ###################################################################

    for b in range(in_tokens.shape[0]):
        for i in range(n_ctx_in - 1):
            print(in_tokens[b, i + 1].to('cpu').detach().numpy().copy(), tokenizer.decode(in_tokens[b, i + 1]), out_alignProbs[b, i])
        print("")

def testOnnx_TextDecoder_StaticLoop(name, model, n_ctx_in: int, n_ctx_out: int, isDynamic: bool):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    #audio_feature = gen_audio_feature(model)
    audio_feature = load_audio_feature(name, 768, model.whisper.device)
    in_tokens = gen_tokens(isMultilingual, tokenizer, n_ctx_in)
    audio_feature_zeros = gen_audio_feature_zeros(model)
    in_tokens_zeros = gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in)

    ###################################################################
    sess_options = onnxruntime.SessionOptions()
    #providers = ['DmlExecutionProvider']
    providers = ['CUDAExecutionProvider']
    #providers = ['CPUExecutionProvider']
    #sess_options.log_severity_level = 0
    #sess_options.log_verbosity_level = 1
    #sess_options.enable_profiling = True
    
    if isDynamic:
        #model_path = f'decoder_sl_a16_1_-1_{n_ctx_out}_1500_int32_{name}.onnx'
        #model_path = f'decoder_sl_a16_1_-1_{n_ctx_out}_1500_int32_{name}_opt_fp16.onnx'
        model_path = f'decoder_sl_a16_pos_1_-1_{n_ctx_out}_1500_int32_{name}_opt_fp16.onnx'
    else:
        model_path = f'decoder_sl_a16_{n_ctx_in}_{n_ctx_out}_{name}_opt_fp16.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_opt_fp16.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_opt.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_smpl_opt16.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_smpl.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}.onnx'

    load_start = time.time()
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    print("Load took:", time.time() - load_start, "s")

    # warm up
    if isDynamic:
        ort_inputs = {
            'in_tokens':  in_tokens_zeros.to('cpu').detach().numpy().copy().astype(np.int32),
            'audio_feature': audio_feature_zeros.to('cpu').detach().numpy().copy().astype(np.float16),
            'offset': np.ones((1,1)).astype(np.int32),
        }
    else:
        ort_inputs = {
            'in_tokens':  in_tokens_zeros.to('cpu').detach().numpy().copy().astype(np.int64),
            'audio_feature': audio_feature_zeros.to('cpu').detach().numpy().copy().astype(np.float16)
        }
    for k in range(3):
        out_tokens = session.run(None, ort_inputs)

    if isDynamic:
        ort_inputs = {
            'in_tokens':  in_tokens.to('cpu').detach().numpy().copy().astype(np.int32),
            'audio_feature': audio_feature.to('cpu').detach().numpy().copy().astype(np.float16),
            'offset': np.ones((1,1)).astype(np.int32) * n_ctx_in
        }
    else:
        ort_inputs = {
            'in_tokens':  in_tokens.to('cpu').detach().numpy().copy().astype(np.int64),
            'audio_feature': audio_feature.to('cpu').detach().numpy().copy().astype(np.float16)
        }

    inference_start = time.time()
    out_tokens = session.run(None, ort_inputs)
    duration = time.time() - inference_start
    print("ONNX RT Inference took:", duration * 1000, "ms")
    print("ave one inferec Static:", duration * 1000 / (len(out_tokens[0][0]) - 1), "ms")
    ###################################################################

    text = tokenizer.decode(out_tokens[0][0])
    print(out_tokens[0][0].dtype)
    print("ONNX RT:", text)

if __name__ == '__main__':
    #cli()
    from __init__ import load_model
    #model_name = "tiny"
    #model_name = "base"
    model_name = "small"
    #model_name = "medium"
    #model_name = "tiny.en"
    #model_name = "base.en"
    #model_name = "small.en"
    #model_name = "medium.en"

    #model = load_model(model_name)
    #args = {}
    #args["language"] = "en"
    ##args["language"] = "ja"
    #result = model.transcribe("tests/jfk.flac")
    #print(result["text"])
    ##result = model.transcribe("tests/MartinLutherKingTrim.wav", **args)
    ##print(result["text"])

    model = load_model(model_name, device="cuda")
    #model = load_model(model_name, device="cpu")

    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 1)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 2)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 4)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 16)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 32)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 2)

    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 3)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 16, 3, False)

    #testOnnx_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 8, False)

    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 3, True)


    #testOnnx_TextDecoder_StaticLoop(model_name, model,  32, 32)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 32, 32, False)

    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 2, False)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 2, True)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 4)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 16)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 32, False)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 32, True)

    #testOnnx_AudioEncoder(model_name, model, 1500, 0)
    #testTorch_AudioEncoder(model_name, model, 1500, 0)

    #testOnnx_TextDecoder_ForcedAlignment(model_name, model, 32, True)
    #testOnnx_TextDecoder_ForcedAlignment(model_name, model, 32, False)
    #testTorch_TextDecoder_ForcedAlignment(model_name, model, 32, 32, False)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 32, 32, False)

    #testTorch_TextDecoder_DynamicLoop(model_name, model, 16, 128)
    #testOnnx_TextDecoder_DynamicLoop(model_name, model, 16)

    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16, True)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16, True)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16, True)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16, False)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16, False)
    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16, False)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 16, 16, False)

    testOnnx_AudioEncoder(model_name, model, 1500, 0, "raw")
    testOnnx_AudioEncoder(model_name, model, 1500, 0, "opt")
    testOnnx_AudioEncoder(model_name, model, 1500, 0, "opt_fp16")
