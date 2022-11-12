from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

import onnx
from onnxsim import simplify

from model import TextDecoder, ResidualAttentionBlock_KvCache

class WhisperSuppresion(nn.Module):
    def __init__(self, isMultilingual: bool):
        super().__init__()
        self.isMultilingual = isMultilingual
        if isMultilingual:
            self.TOKENS_TO_SUPPRESS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362]
            self.END_TRANSCRIPT = 50257
            self.TIMESTAMP_BEGIN = 50364
        else:
            self.TOKENS_TO_SUPPRESS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50359, 50360, 50361]
            self.END_TRANSCRIPT = 50256
            self.TIMESTAMP_BEGIN = 50363

    def select(condition: Tensor, #[n,1], bool
               val_true: Tensor,  #[n,1]
               val_false: Tensor  #[n,1]
               ):

        #vals = torch.cat([val_true, val_false], dim=-1)
        #conditioned_vals = vals * condition

        val = val_true * condition
        val += val_false * torch.logical_not(condition)

        return val

    def forward(self, probs: Tensor, #[1,51865]
                last_token: Tensor, #[1,1]
                penultimate_token: Tensor #[1,1]   
                ):

        # suppressToken
        probs[:, self.TOKENS_TO_SUPPRESS] = -np.inf

        # pair timestamps
        if penultimate_token is None:
            token = torch.argmax(probs[:, :self.TIMESTAMP_BEGIN], dim=-1, keepdim=True)
        else:
            # replace argmax with TopK if beam search is preferable
            token_text_prob, token_text = torch.max(probs[:, :self.END_TRANSCRIPT]                     , dim=-1, keepdim=True) 
            token_spcl_prob, token_spcl = torch.max(probs[:, self.END_TRANSCRIPT:self.TIMESTAMP_BEGIN ], dim=-1, keepdim=True)
            token_time_prob, token_time = torch.max(probs[:, self.TIMESTAMP_BEGIN:]                    , dim=-1, keepdim=True)

            token_text_or_spcl = torch.where(token_text_prob > token_spcl_prob, token_text, token_spcl)
            token_spcl_or_time = torch.where(token_spcl_prob > token_time_prob, token_spcl, token_time)
            token_txt_spcl_tim = torch.where(token_text_prob > token_spcl_prob, 
                                             torch.where(token_text_prob > token_time_prob, 
                                                         token_text,
                                                         token_time
                                             ),
                                             torch.where(token_spcl_prob > token_time_prob, 
                                                         token_spcl,
                                                         token_time
                                             )
                                )

            # if sum of probability over timestamps is above any other token, sample timestamp
            timeProbSum = torch.sum(probs[:,  self.TIMESTAMP_BEGIN:], dim=-1)
            #textProbMax = torch.max(probs[:, :self.TIMESTAMP_BEGIN ], dim=-1)

            token = torch.where(last_token >= self.TIMESTAMP_BEGIN,
                                torch.where(penultimate_token >= self.TIMESTAMP_BEGIN, 
                                            # No time = text or special
                                            token_text_or_spcl, 
                                            # special or time => sum check
                                            torch.where(timeProbSum >= token_text_prob, 
                                                        token_time,
                                                        token_spcl_or_time
                                                        )
                                            ), 
                                # text or special or time => sum check
                                torch.where(timeProbSum >= token_text_prob, 
                                            token_time,
                                            token_txt_spcl_tim
                                            )
                                ) 

            #probs_noTimeStamp = probs
            #probs_noText      = probs
            #probs_noTimeStamp[:,  self.TIMESTAMP_BEGIN:] = -np.inf
            #probs_noText     [:, :self.END_TRANSCRIPT  ] = -np.inf

            #probs = torch.where(last_token >= self.TIMESTAMP_BEGIN,
            #                    torch.where(penultimate_token >= self.TIMESTAMP_BEGIN, 
            #                                probs_noTimeStamp, 
            #                                probs_noText),
            #                    probs)

        return token


class GreedyDecoder(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, isMultilingual: bool):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.suppressor = WhisperSuppresion(isMultilingual)

    def forward(self, x: Tensor, #[1,n_ctx_in,512]
                last_token: Tensor, #[1,1]
                penultimate_token: Tensor #[1,1]
                ):

        x = x[:,-1,:] #greedy
        #[1,512]

        x = self.textDecoder.ln(x) #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same
        #[1,51865]

        probs = F.softmax(logits, dim=-1) #greedy
        #[1,51865]

        token = self.suppressor(probs, last_token, penultimate_token) #supression

        #x = torch.argmax(probs, dim=-1) #greedy
        return token

class TextDecoder_StaticLoop(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, n_ctx_out: int, isMultilingual: bool):
        super().__init__()
        self.textDecoder = in_textDecoder
        #self.n_ctx_in = n_ctx_in
        self.n_ctx_out = n_ctx_out
        self.greedyDecoder = GreedyDecoder(in_textDecoder, isMultilingual)

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block, cacheReturnRule=0)) # return appended self cache

    def forward(self, tokens: Tensor, 
                xa: Tensor
                ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        out_token_list = []

        cross_k_list = []
        cross_v_list = []
        for block in self.textDecoder.blocks:
            if block.cross_attn:
                cross_k_list.append(block.cross_attn.key(xa))
                cross_v_list.append(block.cross_attn.value(xa))

        penultimateToken = None
        lastToken = tokens[:,-1]
        ############ First itr

        mask = self.textDecoder.mask[:tokens.shape[1], :tokens.shape[1]]
        pos_emb_slice = self.textDecoder.positional_embedding[0:tokens.shape[1]] #diff!
        x = self.textDecoder.token_embedding(tokens) + pos_emb_slice #same
        x = x.to(xa.dtype) #same

        # calc self attention while inputing and outputing kv_cache
        i = 0
        self_k_list = [] #[8,1,n_ctx_in,512]
        self_v_list = []
        for block in self.blocks:
            x, self_k_cache_updated, self_v_cache_updated = block(x, 
                                                                  self_k_cache = None, 
                                                                  self_v_cache = None,
                                                                  cross_k = cross_k_list[i], 
                                                                  cross_v = cross_v_list[i], 
                                                                  mask=mask) # diff!
            self_k_list.append(self_k_cache_updated)
            self_v_list.append(self_v_cache_updated)
            i += 1
        
        token = self.greedyDecoder(x, lastToken, penultimateToken) #token=[1,1]
        out_token_list.append(token)

        penultimateToken = lastToken
        lastToken = token
        position = tokens.shape[1]

        ################## loop itr
        mask = None
        for k in range(self.n_ctx_out - 1):
            pos_emb_slice = self.textDecoder.positional_embedding[position] #diff!
            x = self.textDecoder.token_embedding(token) + pos_emb_slice #same
            x = x.to(xa.dtype) #same

            i = 0
            for block in self.blocks:
                x, self_k_cache_updated, self_v_cache_updated = block(x, 
                                                                      self_k_cache = self_k_list[i], 
                                                                      self_v_cache = self_v_list[i],
                                                                      cross_k = cross_k_list[i], 
                                                                      cross_v = cross_v_list[i], 
                                                                      mask=mask) # diff!
                self_k_list[i] = self_k_cache_updated #(1, n_ctx_cache + n_ctx, 512)
                self_v_list[i] = self_v_cache_updated #(1, n_ctx_cache + n_ctx, 512)
                i += 1

            token = self.greedyDecoder(x, lastToken, penultimateToken)
            out_token_list.append(token)

            penultimateToken = lastToken
            lastToken = token
            position = position + 1

        out_tokens = torch.cat(out_token_list, dim=-1)
        return out_tokens

def export_TextDecoder_StaticLoop(name, model, n_ctx_in: int, n_ctx_out: int):
    isMultilingual = not name.endswith('en')
    decoder = TextDecoder_StaticLoop(model.whisper.decoder, n_ctx_out, isMultilingual)
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
    onnx_model = onnx.load(f'{file_onnx}')
    onnx_model_simp, check = simplify(onnx_model)
    onnx.save(onnx_model_simp, f'{file_simp}')


if __name__ == '__main__':
    #cli()
    from __init__ import load_model
    #model_name = "tiny"
    model_name = "base"
    #model_name = "small"
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

    model = load_model(model_name, device="cpu")

    export_TextDecoder_StaticLoop(model_name, model, 32, 64)   # 1500_0