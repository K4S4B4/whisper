from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from model import TextDecoder, ResidualAttentionBlock_KvCache

class WhisperSuppresion(nn.Module):
    def __init__(self, isMultilingual: bool, device):
        super().__init__()
        self.isMultilingual = isMultilingual
        self.device = device
        if isMultilingual:
            self.SYMBOL_TOKENS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362]
            self.END_TRANSCRIPT = 50257
            self.TIMESTAMP_BEGIN = 50364
            self.MAX_TOKEN = 51865
        else:
            self.SYMBOL_TOKENS = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50359, 50360, 50361]
            self.END_TRANSCRIPT = 50256
            self.TIMESTAMP_BEGIN = 50363
            self.MAX_TOKEN = 51864
        self.SUPPRESS_SYMBOLS__ = torch.zeros((1, self.MAX_TOKEN), dtype=torch.float32).to(device)
        self.SUPPRESS_TIMESTAMP = torch.zeros((1, self.MAX_TOKEN), dtype=torch.float32).to(device)
        self.SUPPRESS_ORDINARY_ = torch.zeros((1, self.MAX_TOKEN), dtype=torch.float32).to(device)
        self.EXTRACT_TIMESTAMP_ = torch.zeros((1, self.MAX_TOKEN), dtype=torch.float32).to(device)
        self.SUPPRESS_SYMBOLS__[:, self.SYMBOL_TOKENS   ] = -np.inf
        self.SUPPRESS_TIMESTAMP[:, self.TIMESTAMP_BEGIN:] = -10^15 #-np.inf
        self.SUPPRESS_ORDINARY_[:, :self.END_TRANSCRIPT ] = -10^15 #-np.inf
        self.EXTRACT_TIMESTAMP_[:, :self.TIMESTAMP_BEGIN] = -10^15 #-np.inf

        #[1,51865,4]
        self.SUPPRESS_MATRIX = torch.stack([self.SUPPRESS_SYMBOLS__,
                                            self.SUPPRESS_TIMESTAMP,
                                            self.SUPPRESS_ORDINARY_,
                                            self.EXTRACT_TIMESTAMP_], dim=2)

    def gpu_where(self, 
                  condition: Tensor, #[n,1], bool
                  val_true: Tensor,  #[n,1]
                  val_false: Tensor  #[n,1]
                  ):

        #vals = torch.cat([val_true, val_false], dim=-1)
        #conditioned_vals = vals * condition

        val = val_true * condition
        val += val_false * (~condition)

        return val

    def forward3(self, probs: Tensor, #[n,51865]
                last_token: Tensor, #[n,1]
                penultimate_token: Tensor #[n,1]   
                ):

        if penultimate_token is None:
            probs += self.SUPPRESS_SYMBOLS__
            #token = torch.argmax(probs[:, :self.TIMESTAMP_BEGIN], dim=-1, keepdim=True)
            _, token = torch.topk(probs[:, :self.TIMESTAMP_BEGIN], k=1, dim=-1)

        else:
            #probs += self.SUPPRESS_SYMBOLS__

            token_text_prob, _ = torch.max(probs[:, :self.END_TRANSCRIPT ], dim=-1, keepdim=True) #[n,1]
            timeProbSum        = torch.sum(probs[:, self.TIMESTAMP_BEGIN:], dim=-1, keepdim=True) #[n,1]

            #[n, 4, 1]
            conditions = torch.stack([torch.tensor([[True]], device=self.device),
                                      last_token >= self.TIMESTAMP_BEGIN and penultimate_token >= self.TIMESTAMP_BEGIN,
                                      last_token >= self.TIMESTAMP_BEGIN and penultimate_token <  self.TIMESTAMP_BEGIN,
                                      timeProbSum >= token_text_prob], 
                                     dim=1).to(torch.float32)
            suppression = torch.matmul(self.SUPPRESS_MATRIX, conditions).squeeze(2)
            probs += suppression

            #token = torch.argmax(probs, dim=-1, keepdim=True)
            _, token = torch.topk(probs, k=1, dim=-1)

        return token

    def forward(self, probs: Tensor, #[1,51865]
                last_token: Tensor, #[1,1]
                penultimate_token: Tensor #[1,1]   
                ):

        # suppressToken
        #probs[:, self.TOKENS_TO_SUPPRESS] = -np.inf
        probs += self.SUPPRESS_SYMBOLS__
        
        # pair timestamps
        if penultimate_token is None:
            token = torch.argmax(probs[:, :self.TIMESTAMP_BEGIN], dim=-1, keepdim=True)
        else:
            # replace argmax with TopK if beam search is preferable
            token_text_prob, token_text = torch.max(probs[:, :self.END_TRANSCRIPT]                     , dim=-1, keepdim=True) 
            token_spcl_prob, token_spcl = torch.max(probs[:, self.END_TRANSCRIPT:self.TIMESTAMP_BEGIN ], dim=-1, keepdim=True)
            token_time_prob, token_time = torch.max(probs[:, self.TIMESTAMP_BEGIN:]                    , dim=-1, keepdim=True)

            #token_text_or_spcl = torch.where(token_text_prob > token_spcl_prob, token_text, token_spcl)
            #token_spcl_or_time = torch.where(token_spcl_prob > token_time_prob, token_spcl, token_time)
            #token_txt_spcl_tim = torch.where(token_text_prob > token_spcl_prob, 
            #                                 torch.where(token_text_prob > token_time_prob, 
            #                                             token_text,
            #                                             token_time
            #                                 ),
            #                                 torch.where(token_spcl_prob > token_time_prob, 
            #                                             token_spcl,
            #                                             token_time
            #                                 )
            #                    )

            token_text_or_spcl = self.gpu_where(token_text_prob > token_spcl_prob, token_text, token_spcl)
            token_spcl_or_time = self.gpu_where(token_spcl_prob > token_time_prob, token_spcl, token_time)
            token_txt_spcl_tim = self.gpu_where(token_text_prob > token_spcl_prob, 
                                             self.gpu_where(token_text_prob > token_time_prob, 
                                                         token_text,
                                                         token_time
                                             ),
                                             self.gpu_where(token_spcl_prob > token_time_prob, 
                                                         token_spcl,
                                                         token_time
                                             )
                                )

            # if sum of probability over timestamps is above any other token, sample timestamp
            timeProbSum = torch.sum(probs[:,  self.TIMESTAMP_BEGIN:], dim=-1)
            #textProbMax = torch.max(probs[:, :self.TIMESTAMP_BEGIN ], dim=-1)

            #token = torch.where(last_token >= self.TIMESTAMP_BEGIN,
            #                    torch.where(penultimate_token >= self.TIMESTAMP_BEGIN, 
            #                                # No time = text or special
            #                                token_text_or_spcl, 
            #                                # special or time => sum check
            #                                torch.where(timeProbSum >= token_text_prob, 
            #                                            token_time,
            #                                            token_spcl_or_time
            #                                            )
            #                                ), 
            #                    # text or special or time => sum check
            #                    torch.where(timeProbSum >= token_text_prob, 
            #                                token_time,
            #                                token_txt_spcl_tim
            #                                )
            #                    ) 

            token = self.gpu_where(last_token >= self.TIMESTAMP_BEGIN,
                                self.gpu_where(penultimate_token >= self.TIMESTAMP_BEGIN, 
                                            # No time = text or special
                                            token_text_or_spcl, 
                                            # special or time => sum check
                                            self.gpu_where(timeProbSum >= token_text_prob, 
                                                        token_time,
                                                        token_spcl_or_time
                                                        )
                                            ), 
                                # text or special or time => sum check
                                self.gpu_where(timeProbSum >= token_text_prob, 
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

class WhisperSuppresion_staticLoop(WhisperSuppresion):
    def forward(self, probs: Tensor, #[n,51865]
                last_token: Tensor, #[n,1]
                penultimate_token: Tensor #[n,1]   
                ):

        if penultimate_token is None:
            probs += self.SUPPRESS_SYMBOLS__
            #token = torch.argmax(probs[:, :self.TIMESTAMP_BEGIN], dim=-1, keepdim=True)
            _, token = torch.topk(probs, k=1, dim=-1)
            print(_, token )

        else:
            textProbMax, _ = torch.max(probs[:, :self.END_TRANSCRIPT ], dim=-1, keepdim=True) #[n,1]
            timeProbSum    = torch.sum(probs[:, self.TIMESTAMP_BEGIN:], dim=-1, keepdim=True) #[n,1]

            is_last_timestamp = last_token >= self.TIMESTAMP_BEGIN
            is_penultimate_timestamp = penultimate_token >= self.TIMESTAMP_BEGIN
            probs += self.SUPPRESS_SYMBOLS__
            probs += self.SUPPRESS_TIMESTAMP * (is_last_timestamp * is_penultimate_timestamp)
            probs += self.SUPPRESS_ORDINARY_ * (is_last_timestamp * ~is_penultimate_timestamp)
            probs += self.EXTRACT_TIMESTAMP_ * (timeProbSum >= textProbMax)
            #token = torch.argmax(probs, dim=-1, keepdim=True)
            _, token = torch.topk(probs, k=1, dim=-1)
            print(_, token )

        return token

class GreedyDecoder(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, isMultilingual: bool):
        super().__init__()
        self.textDecoder = in_textDecoder
        #self.suppressor = WhisperSuppresion(isMultilingual, next(in_textDecoder.parameters()).device)
        self.suppressor = WhisperSuppresion_staticLoop(isMultilingual, next(in_textDecoder.parameters()).device)

    def getProbs(self, x: Tensor): 

        #[1,n_ctx_in,512]
        x = x[:,-1,:] #greedy
        #[1,512]

        x = self.textDecoder.ln(x) #same
        #logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)) #same
        #[1,51865]

        probs = F.softmax(logits, dim=-1) #greedy
        #[1,51865]

        return probs

    def forward(self, x: Tensor, #[1,n_ctx_in,512]
                last_token: Tensor, #[1,1]
                penultimate_token: Tensor #[1,1]
                ):
        #[1,n_ctx_in,512]
        probs = self.getProbs(x)
        #[1,51865]
        token = self.suppressor(probs, last_token, penultimate_token) #supression
        #[1,1]
        return token

class TextDecoder_StaticLoop(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, n_ctx_out: int, isMultilingual: bool, makeOnnxAttentionPastPresent: bool, ioInt: int):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.n_head = in_textDecoder.blocks[0].cross_attn.n_head
        self.scale2 = in_textDecoder.blocks[0].cross_attn.scale2
        #self.n_ctx_in = n_ctx_in
        self.n_ctx_out = n_ctx_out
        self.greedyDecoder = GreedyDecoder(in_textDecoder, isMultilingual)
        self.makeOnnxAttentionPastPresent = makeOnnxAttentionPastPresent

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block, cacheReturnRule=0))

        self.ioInt = ioInt

    def forward(self, tokens: Tensor, 
                xa: Tensor
                ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        xa = xa.float()

        out_token_list = []

        cross_k_list = []
        cross_v_list = []
        for block in self.textDecoder.blocks:
            if block.cross_attn:
                #cross_k_list.append(block.cross_attn.key(xa))
                #cross_v_list.append(block.cross_attn.value(xa))
                cross_k = block.cross_attn.key(xa)
                cross_v = block.cross_attn.value(xa)
                cross_k = cross_k.view(*cross_k.shape[:2], self.n_head, 64).permute(0, 2, 3, 1) * self.scale2
                cross_v = cross_v.view(*cross_v.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)
                cross_k_list.append(cross_k)
                cross_v_list.append(cross_v)
        #n_layer_cross_k = torch.stack(cross_k_list)
        #n_layer_cross_v = torch.stack(cross_v_list)
        n_layer_cross_k = cross_k_list
        n_layer_cross_v = cross_v_list

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
                                                                  cross_k = n_layer_cross_k[i], 
                                                                  cross_v = n_layer_cross_v[i], 
                                                                  mask=mask) # diff!

            if self.makeOnnxAttentionPastPresent: # ONNX Attention Node I/O = past/present
                k_t = self_k_cache_updated.view(*self_k_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
                v_t = self_v_cache_updated.view(*self_v_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
                self_kv_stack = torch.stack([k_t, v_t], dim=0)
                k_t = self_kv_stack[0].permute(0, 2, 1, 3)
                v_t = self_kv_stack[1].permute(0, 2, 1, 3)
                self_k_cache_updated = torch.reshape(k_t, (*k_t.shape[:2], self.n_head * 64))
                self_v_cache_updated = torch.reshape(v_t, (*v_t.shape[:2], self.n_head * 64))
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
                                                                      cross_k = n_layer_cross_k[i], 
                                                                      cross_v = n_layer_cross_v[i], 
                                                                      mask=mask) # diff!
                if self.makeOnnxAttentionPastPresent: # ONNX Attention Node I/O = past/present
                    k_t = self_k_cache_updated.view(*self_k_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
                    v_t = self_v_cache_updated.view(*self_v_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
                    self_kv_stack = torch.stack([k_t, v_t], dim=0)
                    k_t = self_kv_stack[0].permute(0, 2, 1, 3)
                    v_t = self_kv_stack[1].permute(0, 2, 1, 3)
                    self_k_cache_updated = torch.reshape(k_t, (*k_t.shape[:2], self.n_head * 64))
                    self_v_cache_updated = torch.reshape(v_t, (*v_t.shape[:2], self.n_head * 64))
                self_k_list[i] = self_k_cache_updated #(1, n_ctx_cache + n_ctx, 512)
                self_v_list[i] = self_v_cache_updated #(1, n_ctx_cache + n_ctx, 512)
                i += 1

            token = self.greedyDecoder(x, lastToken, penultimateToken)
            out_token_list.append(token)

            penultimateToken = lastToken
            lastToken = token
            position = position + 1

        out_tokens = torch.cat(out_token_list, dim=-1)

        if self.ioInt == 32:
            out_tokens.to(torch.int32)

        return out_tokens

class TextDecoder_ForcedAlignment(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, n_ctx_out: int, isMultilingual: bool, makeOnnxAttentionPastPresent: bool):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.n_head = in_textDecoder.blocks[0].cross_attn.n_head
        self.scale2 = in_textDecoder.blocks[0].cross_attn.scale2
        #self.n_ctx_in = n_ctx_in
        self.n_ctx_out = n_ctx_out
        self.suppression = WhisperSuppresion(isMultilingual, next(in_textDecoder.parameters()).device)
        self.makeOnnxAttentionPastPresent = makeOnnxAttentionPastPresent

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block, cacheReturnRule=0)) 

    def forward(self, tokens: Tensor, #[n, n_ctx_in]
                xa: Tensor
                ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        xa = xa.float()

        out_token_list = []

        cross_k_list = []
        cross_v_list = []
        for block in self.textDecoder.blocks:
            if block.cross_attn:
                #cross_k_list.append(block.cross_attn.key(xa))
                #cross_v_list.append(block.cross_attn.value(xa))
                cross_k = block.cross_attn.key(xa)
                cross_v = block.cross_attn.value(xa)
                cross_k = cross_k.view(*cross_k.shape[:2], self.n_head, 64).permute(0, 2, 3, 1) * self.scale2
                cross_v = cross_v.view(*cross_v.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)
                cross_k_list.append(cross_k)
                cross_v_list.append(cross_v)
        #n_layer_cross_k = torch.stack(cross_k_list)
        #n_layer_cross_v = torch.stack(cross_v_list)
        n_layer_cross_k = cross_k_list
        n_layer_cross_v = cross_v_list

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
                                                                  cross_k = n_layer_cross_k[i], 
                                                                  cross_v = n_layer_cross_v[i], 
                                                                  mask=mask) # diff!

            if self.makeOnnxAttentionPastPresent: # ONNX Attention Node I/O = past/present
                k_t = self_k_cache_updated.view(*self_k_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
                v_t = self_v_cache_updated.view(*self_v_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
                self_kv_stack = torch.stack([k_t, v_t], dim=0)
                k_t = self_kv_stack[0].permute(0, 2, 1, 3)
                v_t = self_kv_stack[1].permute(0, 2, 1, 3)
                self_k_cache_updated = torch.reshape(k_t, (*k_t.shape[:2], self.n_head * 64))
                self_v_cache_updated = torch.reshape(v_t, (*v_t.shape[:2], self.n_head * 64))
            self_k_list.append(self_k_cache_updated)
            self_v_list.append(self_v_cache_updated)
            i += 1
        
        ###########################################
        # diff from here
        ###########################################

        x = self.textDecoder.ln(x) #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same
        #[n, n_ctx_in, 51865]

        probs = F.softmax(logits, dim=-1)
        #[n, n_ctx_in, 51865]

        if tokens.dtype == torch.int32:
            forcedProbs = torch.gather(probs[:, 0:-1, :], 2, tokens[:, 1:].unsqueeze(-1).to(torch.int64))
        else:
            forcedProbs = torch.gather(probs[:, 0:-1, :], 2, tokens[:, 1:].unsqueeze(-1))

        return forcedProbs

class WhisperSuppresion_dynamicLoop(WhisperSuppresion):
    def forward(self, 
                probs: Tensor, #[n,51865]
                ):

        textProbMax, _ = torch.max(probs[:, :self.END_TRANSCRIPT ], dim=-1, keepdim=True) #[n,1]
        timeProbSum    = torch.sum(probs[:, self.TIMESTAMP_BEGIN:], dim=-1, keepdim=True) #[n,1]

        probs += self.SUPPRESS_SYMBOLS__
        probs += self.EXTRACT_TIMESTAMP_ * (timeProbSum > textProbMax)
        _, token = torch.topk(probs, k=1, dim=-1)

        return token

class GreedyDecoder_dynamicLoop(GreedyDecoder):
    def __init__(self, in_textDecoder: TextDecoder, isMultilingual: bool):
        super().__init__(in_textDecoder, isMultilingual)
        self.suppressor = WhisperSuppresion_dynamicLoop(isMultilingual, next(in_textDecoder.parameters()).device)

    def forward(self, x: Tensor, #[1,n_ctx_in,512]
                ):
        probs = self.getProbs(x)
        token = self.suppressor(probs) #supression
        return token

class CrossKvCalculator(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.scale2 = in_textDecoder.blocks[0].cross_attn.scale2

    def forward(self, xa: Tensor):
        cross_k_list = []
        cross_v_list = []
        for block in self.textDecoder.blocks:
            if block.cross_attn:
                cross_k = block.cross_attn.key(xa)
                cross_v = block.cross_attn.value(xa)
                cross_k = cross_k.view(*cross_k.shape[:2], block.cross_attn.n_head, 64).permute(0, 2, 3, 1) * self.scale2
                cross_v = cross_v.view(*cross_v.shape[:2], block.cross_attn.n_head, 64).permute(0, 2, 1, 3)
                cross_k_list.append(cross_k)
                cross_v_list.append(cross_v)
        return cross_k_list, cross_v_list

class TextDecoder_dynamicLoop(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, n_ctx_out: int, isMultilingual: bool, n_layer: int):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.n_head = in_textDecoder.blocks[0].cross_attn.n_head
        self.scale2 = in_textDecoder.blocks[0].cross_attn.scale2
        #self.n_ctx_in = n_ctx_in
        self.n_layer = n_layer
        self.n_ctx_out = n_ctx_out
        self.device = next(in_textDecoder.parameters()).device
        self.greedyDecoder = GreedyDecoder_dynamicLoop(in_textDecoder, isMultilingual)
        self.suppression = WhisperSuppresion_dynamicLoop(isMultilingual, self.device)
        self.crossKvCalculator = CrossKvCalculator(in_textDecoder)

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block, cacheReturnRule=0)) #self_kv_cache shape = (1, n_ctx_cache + n_ctx, 512)

        self.REPEAT_MAX = 8

    #def calcCrossKv(self, 
    #               xa: Tensor
    #               ):
    #    #xa = xa.float()
    #    cross_k_list = []
    #    cross_v_list = []
    #    for block in self.textDecoder.blocks:
    #        if block.cross_attn:
    #            cross_k = block.cross_attn.key(xa)
    #            cross_v = block.cross_attn.value(xa)
    #            cross_k = cross_k.view(*cross_k.shape[:2], self.n_head, 64).permute(0, 2, 3, 1) * self.scale2
    #            cross_v = cross_v.view(*cross_v.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)
    #            cross_k_list.append(cross_k)
    #            cross_v_list.append(cross_v)
    #    return cross_k_list, cross_v_list

    def loopBody(self, 
                 trip_count: Tensor, #int64
                 in_cond: Tensor, #bool
                 in_tokens: Tensor, #int32[b,t]
                 in_positions: Tensor, #int32[b,t]
                 in_token_history: Tensor, #int64[b,h]
                 in_repeat_count: Tensor, #int64
                 in_self_kv_list, #float16[2,b,n_head,n_ctx_total,64] list
                 n_layer_cross_k, #float16[  b,n_head,64,1500]
                 n_layer_cross_v, #float16[  b,n_head,64,1500]
                 ):
        #in_tokens = in_tokens.to(torch.int64)
        #in_positions = in_positions.to(torch.int64)

        #mask = self.textDecoder.mask[:in_tokens.shape[1], :in_tokens.shape[1]]
        mask = None
        if in_positions.dtype == torch.int64:
            pos_emb_slice = self.textDecoder.positional_embedding[in_positions] #diff!
        else:
            pos_emb_slice = self.textDecoder.positional_embedding[in_positions.to(torch.int64)] #diff!
        x = self.textDecoder.token_embedding(in_tokens) + pos_emb_slice #same
        #x = x.to(xa.dtype) #same

        out_self_kv_list = []
        i = 0
        for block in self.blocks: 
            if in_self_kv_list is None:
                self_k = None
                self_v = None
            else:
                self_k = in_self_kv_list[i][0] #float16[b,n_head,n_ctx_cache,64]
                self_v = in_self_kv_list[i][1] #float16[b,n_head,n_ctx_cache,64]
                self_k = self_k.permute(0, 2, 1, 3) #float16[b,n_ctx_cache,n_head,64]
                self_v = self_v.permute(0, 2, 1, 3) #float16[b,n_ctx_cache,n_head,64]
                self_k = torch.reshape(self_k, (*self_k.shape[:2], self.n_head * 64)) #float16[b,n_ctx_cache,512]
                self_v = torch.reshape(self_v, (*self_v.shape[:2], self.n_head * 64)) #float16[b,n_ctx_cache,512]

            x, self_k_cache_updated, self_v_cache_updated = block(x, 
                                                                  self_k_cache = self_k, 
                                                                  self_v_cache = self_v,
                                                                  cross_k = n_layer_cross_k[i], 
                                                                  cross_v = n_layer_cross_v[i], 
                                                                  mask=mask) # diff!

            k_t = self_k_cache_updated.view(*self_k_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
            v_t = self_v_cache_updated.view(*self_v_cache_updated.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)#(1, 8, n_ctx_cache + n_ctx, 64)
            self_kv_stack = torch.stack([k_t, v_t], dim=0) #[2,b,n_head,n_ctx_total,64]
            out_self_kv_list.append(self_kv_stack)         #[2,b,n_head,n_ctx_total,64] list
            i += 1

        out_token = self.greedyDecoder(x)
        out_token_history = torch.cat([in_token_history, out_token], dim=-1)
        out_positions = in_positions[:,-1] + torch.ones([1,1], dtype=torch.int32).to(self.device)

        #impl repeat termination condition
        hasToken = (in_token_history == out_token)
        #out_repeat_count = in_repeat_count + torch.sum(hasToken, dim=-1, keepdim=True)
        out_repeat_count = in_repeat_count + torch.any(hasToken, dim=-1, keepdim=True)
        out_cond = (out_token < self.suppression.END_TRANSCRIPT) * (out_repeat_count < self.REPEAT_MAX)
        if out_cond.shape[0] == 1 and out_cond.shape[1] == 1:
            out_cond = out_cond.squeeze()
        else:
            out_cond = torch.any(out_cond)

        #debug
        if out_cond == False:
            out_cond = False

        return (
            out_cond, 
            out_token,
            out_positions,
            out_token_history, #int64[b,h+1]
            out_repeat_count,  #int64[b,1]
            out_self_kv_list,
            )

    def preprocess(self, 
            in_tokens: Tensor, #[n, n_ctx_in]
            #in_positions: Tensor, #[n, n_ctx_in]
            xa: Tensor
            ):

        trip_count = torch.tensor(self.n_ctx_out, dtype=torch.int64).to(self.device) #max 128 loop
        in_cond = torch.tensor(True, dtype=torch.bool).to(self.device)
        #in_positions = torch.arange(0, in_tokens.shape[1]).unsqueeze(0).to(self.device)
        in_token_history = in_tokens[:,0:1]
        in_repeat_count = torch.zeros([1,1], dtype=torch.int64).to(self.device)

        self_kv_list = []
        for i in range(self.n_layer):
            self_kv_list.append(torch.zeros([2, 1, self.n_head, 0, 64], dtype=torch.float32).to(self.device))
        #n_layer_cross_k, n_layer_cross_v = self.calcCrossKv(xa)
        n_layer_cross_k, n_layer_cross_v = self.crossKvCalculator(xa)

        return (
                trip_count, in_cond, #in_tokens, in_positions, 
                in_token_history, in_repeat_count,
                self_kv_list,
                n_layer_cross_k,
                n_layer_cross_v,
        )

    def run(self, 
            in_tokens: Tensor, #[n, n_ctx_in]
            in_positions: Tensor, #[n, n_ctx_in]
            xa: Tensor
            ):

        #trip_count = torch.tensor(128, dtype=torch.int64) #max 128 loop
        #in_cond = torch.tensor(True, dtype=torch.bool)
        #in_positions = torch.arange(0, in_tokens.shape[1]).unsqueeze(0)
        #in_self_kv_list = None
        #n_layer_cross_k, n_layer_cross_v = self.calcCrossKv(xa)

        (
            #out_token_scan,
            trip_count, in_cond, 
            in_token_history, in_repeat_count,
            in_self_kv_list,
            n_layer_cross_k,
            n_layer_cross_v,
        ) = self.preprocess(in_tokens, xa)

        #out_token_scan_list = []
        for k in range(self.n_ctx_out):
            (
            out_cond, 
            out_token,
            out_positions,
            out_token_history, 
            out_repeat_count, 
            out_self_kv_list,
            ) = self.loopBody(
            trip_count, 
            in_cond, 
            in_tokens, 
            in_positions, 
            in_token_history, 
            in_repeat_count,
            in_self_kv_list,
            n_layer_cross_k, 
            n_layer_cross_v 
            )

            #out_token_scan_list.append(out_token) #[1,1] list
            if not out_cond:
                break
            
            in_tokens = out_token
            in_positions = out_positions
            in_token_history = out_token_history
            in_repeat_count  = out_repeat_count
            in_self_kv_list = out_self_kv_list

        #out_token_scan = torch.stack(out_token_scan_list) #[n,1,1]
        #return out_token_scan
        return out_token_history

