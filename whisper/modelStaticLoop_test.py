import os.path
import numpy as np
import torch
import onnxruntime
import time

import cv2

from audio import load_audio, log_mel_spectrogram, SAMPLE_RATE
from modelStaticLoop import TextDecoder_StaticLoop
from tokenizer import get_tokenizer

def gen_audio_feature_zeros(model):
    return torch.zeros((1,1500,model.whisper.dims.n_text_state), dtype=torch.float32).to(model.whisper.device)

def gen_mel_zeros(model):
    return torch.zeros((1,3000,80), dtype=torch.float32).to(model.whisper.device)

def gen_mel(model):
    audio_path = "tests/jfk.flac"
    mel = log_mel_spectrogram(audio_path).unsqueeze(0)
    mel_t = mel.permute(0, 2, 1)

    n_pad_before = 500
    n_pad = 3000 - mel_t.shape[1] - n_pad_before
    pad_before = torch.zeros(1,n_pad_before,80)
    pad = torch.zeros(1,n_pad,80)
    mel_t = torch.cat([pad_before, mel_t, pad], dim=1).to(model.whisper.device)
    return mel_t

def gen_audio_feature(model):
    mel_t = gen_mel(model)
    encoder = model.whisper.encoder
    return encoder(mel_t)

def gen_tokens_zeros(isMultilingual, tokenizer, n_ctx_in: int):
    return torch.zeros((1, n_ctx_in), dtype=torch.int64)

def gen_tokens(isMultilingual, tokenizer, n_ctx_in: int):
    in_tokens = torch.ones((1, n_ctx_in), dtype=torch.int64)
    #if isMultilingual:
    #    in_tokens[:,n_ctx_in-1] = tokenizer.timestamp_begin
    #    in_tokens[:,n_ctx_in-2] = 50359
    #    in_tokens[:,n_ctx_in-3] = 50259
    #    in_tokens[:,n_ctx_in-4] = tokenizer.sot
    #    in_tokens[:,n_ctx_in-5] = tokenizer.timestamp_begin + 1499
    #    in_tokens[:,n_ctx_in-6] = tokenizer.timestamp_begin + 1499
    #    in_tokens[:,0] = tokenizer.sot_prev
    #else:
    #    in_tokens[:,n_ctx_in-1] = tokenizer.timestamp_begin
    #    in_tokens[:,n_ctx_in-2] = tokenizer.sot
    #    in_tokens[:,n_ctx_in-3] = tokenizer.timestamp_begin + 1499
    #    in_tokens[:,n_ctx_in-4] = tokenizer.timestamp_begin + 1499
    #    in_tokens[:,0] = tokenizer.sot_prev

    if isMultilingual:
        #in_tokens[:,n_ctx_in-1] = tokenizer.timestamp_begin
        in_tokens[:,n_ctx_in-1] = 50359
        in_tokens[:,n_ctx_in-2] = 50259
        in_tokens[:,n_ctx_in-3] = tokenizer.sot
        in_tokens[:,n_ctx_in-4] = tokenizer.timestamp_begin + 1499
        in_tokens[:,n_ctx_in-5] = tokenizer.timestamp_begin + 1499
        in_tokens[:,0] = tokenizer.sot_prev
    else:
        #in_tokens[:,n_ctx_in-1] = tokenizer.timestamp_begin
        in_tokens[:,n_ctx_in-1] = tokenizer.sot
        in_tokens[:,n_ctx_in-2] = tokenizer.timestamp_begin + 1499
        in_tokens[:,n_ctx_in-3] = tokenizer.timestamp_begin + 1499
        in_tokens[:,0] = tokenizer.sot_prev
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

def testOnnx_AudioEncoder(name, model, n_ctx_in: int, n_ctx_out: int):
    mel_t_zeros = gen_mel_zeros(model)
    mel_t = gen_mel(model)

    ###################################################################
    sess_options = onnxruntime.SessionOptions()
    #providers = ['DmlExecutionProvider']
    providers = ['CUDAExecutionProvider']
    #providers = ['CPUExecutionProvider']
    #sess_options.log_severity_level = 0
    #sess_options.log_verbosity_level = 1
    #sess_options.enable_profiling = True
    
    #model_path = f'encoder_org_{n_ctx_in}_{n_ctx_out}_{name}_smpl.onnx'
    #model_path = f'encoder_org_{n_ctx_in}_{n_ctx_out}_{name}.onnx'
    #model_path = f'encoder_org_{n_ctx_in}_{n_ctx_out}_{name}_opt.onnx'
    model_path = f'encoder_org_{n_ctx_in}_{n_ctx_out}_{name}_opt_fp16.onnx'

    load_start = time.time()
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    print("Load took:", time.time() - load_start, "s")

    # warm up
    ort_inputs = {
        'mel':  mel_t_zeros.to('cpu').detach().numpy().copy().astype(np.float32)
    }
    for k in range(5):
        out_audio_feature = session.run(None, ort_inputs)

    ort_inputs = {
        'mel':  mel_t.to('cpu').detach().numpy().copy().astype(np.float32)
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

    decoder = TextDecoder_StaticLoop(model.whisper.decoder, n_ctx_out, isMultilingual, makeOnnxAttentionPastPresent)

    # warm up
    for k in range(5):
        out_tokens = decoder(in_tokens_zeros, audio_feature_zeros)

    inference_start = time.time()
    out_tokens = decoder(in_tokens, audio_feature)
    print("PyTorch Inference took:", (time.time() - inference_start) * 1000, "ms")

    out_token_list = []
    for i in range(n_ctx_out):
        out_token_list.append(out_tokens[0, i])
    text = tokenizer.decode(out_token_list)

    print("PyTorch:", out_token_list[0] - tokenizer.timestamp_begin)
    print("PyTorch:", text)

def testOnnx_TextDecoder_StaticLoop(name, model, n_ctx_in: int, n_ctx_out: int):
    isMultilingual = not name.endswith('en')
    tokenizer = get_tokenizer(multilingual=isMultilingual)
    audio_feature = gen_audio_feature(model)
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
    
    model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_opt_fp16.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_opt.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_smpl_opt16.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}_smpl.onnx'
    #model_path = f'decoder_staticLoop_{n_ctx_in}_{n_ctx_out}_{name}.onnx'

    load_start = time.time()
    session = onnxruntime.InferenceSession(model_path, sess_options, providers)
    print("Load took:", time.time() - load_start, "s")

    # warm up
    ort_inputs = {
        'in_tokens':  in_tokens_zeros.to('cpu').detach().numpy().copy().astype(np.int64),
        'audio_feature': audio_feature_zeros.to('cpu').detach().numpy().copy().astype(np.float32)
    }
    for k in range(5):
        out_tokens = session.run(None, ort_inputs)

    ort_inputs = {
        'in_tokens':  in_tokens.to('cpu').detach().numpy().copy().astype(np.int64),
        'audio_feature': audio_feature.to('cpu').detach().numpy().copy().astype(np.float32)
    }
    inference_start = time.time()
    out_tokens = session.run(None, ort_inputs)
    print("ONNX RT Inference took:", (time.time() - inference_start) * 1000, "ms")
    ###################################################################

    text = tokenizer.decode(out_tokens[0][0])

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

    #testOnnx_TextDecoder_StaticLoop(model_name, model, 16, 16)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 16, 16, False)

    #testOnnx_TextDecoder_StaticLoop(model_name, model,  32, 32)
    testTorch_TextDecoder_StaticLoop(model_name, model, 32, 32, False)

    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 2, False)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 2, True)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 4)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 16)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 32, False)
    #testTorch_TextDecoder_StaticLoop(model_name, model, 8, 32, True)

    #testOnnx_AudioEncoder(model_name, model, 1500, 0)
    #testTorch_AudioEncoder(model_name, model, 1500, 0)
