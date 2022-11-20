from cmath import inf
from dataclasses import dataclass
from pickle import NONE
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from transcribe import transcribe as transcribe_function
from decoding import detect_language as detect_language_function, decode as decode_function

import onnx
from onnxsim import simplify
import cv2

#FOR_ONNX_EXPORT: bool = False
FOR_ONNX_EXPORT: bool = True

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

def getDims(model_name):
    dims = ModelDimensions()
    if model_name.startswith("tiny"):
        dims.n_audio_state = 384
        dims.n_audio_layer = 8
    if model_name.startswith("base"):
        dims.n_audio_state = 512
        dims.n_audio_layer = 12
    if model_name.startswith("small"):
        dims.n_audio_state = 768
        dims.n_audio_layer = 24
    if model_name.startswith("medium"):
        dims.n_audio_state = 1024
        dims.n_audio_layer = 48
    dims.n_mels = 3000
    dims.n_audio_ctx = 1500
    dims.n_text_state = dims.n_audio_state
    dims.n_text_layer = dims.n_audio_layer
    dims.n_text_head = int(dims.n_text_state / 64)
    dims.n_audio_head = int(dims.n_audio_state / 64)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        if FOR_ONNX_EXPORT:
            return super().forward(x)
        else:
            return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.scale = (n_state // n_head) ** -0.25
        self.scale2 = (n_state // n_head) ** -0.5

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache.get(self.key, self.key(xa))
            v = kv_cache.get(self.value, self.value(xa))

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape

        #q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        #k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        #v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        # this gives the same result. Note that values are selected so that (n_state / n_head) = 64
        q = q.view(*q.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)# * self.scale
        k = k.view(*k.shape[:2], self.n_head, 64).permute(0, 2, 3, 1)# * self.scale
        v = v.view(*v.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)

        qk = q @ k * self.scale2
        if mask is not None:
            #qk = qk + mask[:n_ctx, :n_ctx] #これが出てくるのはSelfAttentionのとき。queryのTokenより未来のTokenのkeyを問い合わせてるときは-Infにしてしまう。いや、これ要るのか？未来情報で過去を改善できそうなのだが。
            qk = qk + mask #specify as dynamic input tensor

            #n_ctx_cache = k.shape[-1] - n_ctx
            #_, _, n_ctx, n_ctx_cache_plus_n_ctx = qk.shape

            #mask0 = torch.zeros((n_ctx, n_ctx_cache), dtype=q.dtype).to(q.device);
            #mask_cat = torch.cat((mask0, mask[:n_ctx, :n_ctx]), dim=1) #(n_ctx, n_ctx_cache + n_ctx)
            #qk = qk + mask_cat

        #global FOR_ONNX_EXPORT
        if FOR_ONNX_EXPORT:
            w = F.softmax(qk, dim=-1)
        else:
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)

        #return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        x = (w @ v).permute(0, 2, 1, 3)
        return torch.reshape(x, (1, x.shape[1], self.n_head * 64))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        #xnum = x.squeeze().to('cpu').detach().numpy().copy().astype(np.float32)
        #xnum = cv2.resize(xnum, (1500, 320))
        #cv2.imshow("EncBeferConv", xnum)
        #cv2.waitKey(1)

        x = x.permute(0, 2, 1) #diff

        #x = torch.clamp(x, min=1e-10).log10() #(80,1100)
        x = torch.maximum(x, x.max() - 8.0) #(80,1100)
        x = (x + 4.0) / 4.0 #(80,1100)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        #xnum = x.squeeze().to('cpu').detach().numpy().copy().astype(np.float32)
        #cv2.imshow("EncAfterConv", xnum)
        #cv2.waitKey(0)

        x = x.permute(0, 2, 1)

        #global FOR_ONNX_EXPORT
        if FOR_ONNX_EXPORT:
            x = x + self.positional_embedding
        else:
            assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder.forward(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder.forward(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

class MultiHeadAttention_CrossKvCache(nn.Module):
    def __init__(self, in_multiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = in_multiHeadAttention

    def qkv_attention_cross(self, q: Tensor, k_t: Tensor, v_t: Tensor, mask: Optional[Tensor] = None):
        # this gives the same result. Note that values are selected so that (n_state / n_head) = 64
        q = q.view(*q.shape[:2], self.multiHeadAttention.n_head, 64).permute(0, 2, 1, 3)# * self.scale
        k = k_t
        v = v_t

        qk = q @ k #* self.multiHeadAttention.scale2

        #global FOR_ONNX_EXPORT
        if FOR_ONNX_EXPORT:
            w = F.softmax(qk, dim=-1)
        else:
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)

        #return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        x = (w @ v).permute(0, 2, 1, 3)
        return torch.reshape(x, (*x.shape[:2], self.multiHeadAttention.n_head * 64))

    def forward(
        self,
        x: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        if cross_v.shape[-1] == 64:
            # if k, v reshape and permute are precomputed
            wv = self.qkv_attention_cross(q, cross_k, cross_v, mask)
        else:
            wv = self.multiHeadAttention.qkv_attention(q, cross_k, cross_v, mask)
        return self.multiHeadAttention.out(wv)

class MultiHeadAttention_SelfKvCache(nn.Module):
    def __init__(self, in_multiHeadAttention: MultiHeadAttention, in_cacheReturnRule: int):
        super().__init__()
        self.multiHeadAttention = in_multiHeadAttention
        self.cacheReturnRule = in_cacheReturnRule

    def qkv_attention_self(self, q: Tensor, k_t: Tensor, v_t: Tensor, mask: Optional[Tensor] = None):
        # this gives the same result. Note that values are selected so that (n_state / n_head) = 64
        q = q.view(*q.shape[:2], self.multiHeadAttention.n_head, 64).permute(0, 2, 1, 3)
        k = k_t.permute(0, 1, 3, 2)
        v = v_t

        qk = q @ k * self.multiHeadAttention.scale2

        #global FOR_ONNX_EXPORT
        if FOR_ONNX_EXPORT:
            w = F.softmax(qk, dim=-1)
        else:
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)

        x = (w @ v).permute(0, 2, 1, 3)
        return torch.reshape(x, (*x.shape[:2], self.multiHeadAttention.n_head * 64))

    def forward(
        self,
        x: Tensor,
        self_k_cache: Optional[Tensor] = None, #(1, n_ctx_cache, 512)
        self_v_cache: Optional[Tensor] = None, #(1, n_ctx_cache, 512)
        positions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        n_ctx = x.shape[1]

        q = self.multiHeadAttention.query(x)
        k = self.multiHeadAttention.key(x)   #(1, n_ctx, 512)
        v = self.multiHeadAttention.value(x) #(1, n_ctx, 512)

        if self.cacheReturnRule == 3: #return present (1, 8, n_ctx, 64) for Attention node of ONNX contrib com.microsft
            k_t = k.view(*k.shape[:2], self.multiHeadAttention.n_head, 64).permute(0, 2, 1, 3)
            v_t = v.view(*v.shape[:2], self.multiHeadAttention.n_head, 64).permute(0, 2, 1, 3)
            if self_k_cache is not None:
                k_t = torch.cat((self_k_cache, k_t), 2) #(1, 8, n_ctx_cache + n_ctx, 64)
                v_t = torch.cat((self_v_cache, v_t), 2) #(1, 8, n_ctx_cache + n_ctx, 64)
            wv = self.qkv_attention_self(q, k_t, v_t, mask)
            return self.multiHeadAttention.out(wv), k_t, v_t

        if self_k_cache is None:
            wv = self.multiHeadAttention.qkv_attention(q, k, v, mask)
            return self.multiHeadAttention.out(wv), k, v

        if positions is None:
            k_before_append = k
            v_before_append = v

            if self_k_cache is not None:
                k = torch.cat((self_k_cache, k), 1) #(1, n_ctx_cache + n_ctx, 512)
                v = torch.cat((self_v_cache, v), 1) #(1, n_ctx_cache + n_ctx, 512)

            wv = self.multiHeadAttention.qkv_attention(q, k, v, mask)

            if self.cacheReturnRule == 0: #return just appended cache (n_ctx_cache + n_ctx)

                return self.multiHeadAttention.out(wv), k, v

            elif self.cacheReturnRule == 1: #return appended and shrinked to the size of input cache (n_ctx_cache)

                #DONE shrinked to the size of input cache!!!
                n_ctx_cache = self_k_cache.shape[1]
                self_k_cache_shrinked = k[:,n_ctx:n_ctx+n_ctx_cache,:] #(1, n_ctx_cache + n_ctx, 512) -> (1, n_ctx_cache, 512). Get last (n_ctx_cache)
                self_v_cache_shrinked = v[:,n_ctx:n_ctx+n_ctx_cache,:] #(1, n_ctx_cache + n_ctx, 512) -> (1, n_ctx_cache, 512)
            
                return self.multiHeadAttention.out(wv), self_k_cache_shrinked, self_v_cache_shrinked

            elif self.cacheReturnRule == 2: #return only new cache (n_ctx)

                return self.multiHeadAttention.out(wv), k_before_append, v_before_append

        else: # similar to rule=1, but not append to the last. It overwrites the specified positions of the cache and return whole cache (n_ctx_cache). Meant to use for Encoder with static i/o of kv_cache (1,1500,512) if DirectML input with GPU memory achieved(Plan1)
            self_k_cache[:, positions, :] = k
            self_v_cache[:, positions, :] = v
            wv = self.multiHeadAttention.qkv_attention(q, self_k_cache, self_v_cache, mask)
            return self.multiHeadAttention.out(wv), self_k_cache, self_v_cache

class ResidualAttentionBlock_KvCache(nn.Module):
    def __init__(self, in_residualAttentionBlock: ResidualAttentionBlock, cacheReturnRule: int):
        super().__init__()
        self.originalBlock = in_residualAttentionBlock
        self.attn = MultiHeadAttention_SelfKvCache(in_residualAttentionBlock.attn, cacheReturnRule)
        self.cross_attn = MultiHeadAttention_CrossKvCache(in_residualAttentionBlock.cross_attn) if in_residualAttentionBlock.cross_attn else None

    def forward(
        self,
        x: Tensor,
        self_k_cache: Optional[Tensor] = None,
        self_v_cache: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        cross_k: Optional[Tensor] = None,
        cross_v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, positions, mask=mask) #diff!
        x = x + self_attn_x

        if self.cross_attn:
            x = x + self.cross_attn(self.originalBlock.cross_attn_ln(x), cross_k, cross_v) #diff!

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x)) #same as orginal
        return x, self_k_cache_updated, self_v_cache_updated

class AudioEncoder_KvCache(nn.Module):
    def __init__(self, in_audioEncoder: AudioEncoder, in_textDecoder: TextDecoder, in_n_ctx: int, in_n_ctx_cache: int):
        super().__init__()
        self.audioEncoder = in_audioEncoder
        self.textDecoder = in_textDecoder
        self.n_ctx_cache = in_n_ctx_cache #値どこでも使ってないぞい
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.audioEncoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block, cacheReturnRule=2))

    #def forward(self, x: Tensor, n_layer_self_k_cache: Tensor, n_layer_self_v_cache: Tensor, offset: int):
    def forward(self, 
                x: Tensor, 
                n_layer_self_k_cache: Tensor, 
                n_layer_self_v_cache: Tensor, 
                positions: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        n_layer_self_k_cache : shape = (n_layer, 1, n_ctx_cache, 512)
        n_layer_self_v_cache : shape = (n_layer, 1, n_ctx_cache, 512)
        offset : where the positional_embedding starts for x. [0, 1499]
        return : n_layer_cross_k, n_layer_cross_v : precomputed kv for cross attention from decoder's query. shape = (n_layer, 1, n_ctx, 512)
        return : n_layer_self_k_cache_updated, n_layer_self_v_cache_updated : precomputed kv for self attention from encoder's query. shape = (n_layer, 1, n_ctx_cache + n_ctx, 512)
        """
        print(x.shape)

        # pre process
        x = x.permute(0, 2, 1) #diff

        #x = torch.clamp(x, min=1e-10).log10() #(80,1100)
        x = torch.maximum(x, x.max() - 8.0) #(80,1100)
        x = (x + 4.0) / 4.0 #(80,1100)

        x = F.gelu(self.audioEncoder.conv1(x)) #same
        x = F.gelu(self.audioEncoder.conv2(x)) #same
        x = x.permute(0, 2, 1) #same

        pos_emb = self.audioEncoder.positional_embedding[positions] #diff
        x = (x + pos_emb).to(x.dtype)

        #diff
        # calc self attention while inputing and outputing kv_cache
        i = 0
        self_k_cache_update_list = []
        self_v_cache_update_list = []
        for block in self.blocks:
            x, self_k_cache_updated, self_v_cache_updated = block(x, 
                                                                  self_k_cache = n_layer_self_k_cache[i], 
                                                                  self_v_cache = n_layer_self_v_cache[i]
                                                                  #,positions = positions
                                                                  ) 
            self_k_cache_update_list.append(self_k_cache_updated)
            self_v_cache_update_list.append(self_v_cache_updated)
            i += 1

        n_layer_self_k_cache_updated = torch.stack(self_k_cache_update_list)
        n_layer_self_v_cache_updated = torch.stack(self_v_cache_update_list)

        xa = self.audioEncoder.ln_post(x) #same

        #xnum = xa.squeeze().to('cpu').detach().numpy().copy().astype(np.float32).transpose()
        #cv2.imshow("Enc_xa", xnum)
        #cv2.waitKey(0)

        # pre compute key-value for audio feature
        cross_k_list = []
        cross_v_list = []
        for block in self.textDecoder.blocks:
            if block.cross_attn:
                cross_k_list.append(block.cross_attn.key(xa))
                cross_v_list.append(block.cross_attn.value(xa))

        n_layer_cross_k = torch.stack(cross_k_list)
        n_layer_cross_v = torch.stack(cross_v_list)

        #ONNX export
        #global FOR_ONNX_EXPORT
        if FOR_ONNX_EXPORT:
            return cross_k_list, cross_v_list, self_k_cache_update_list, self_v_cache_update_list
        else:
            return n_layer_cross_k, n_layer_cross_v, n_layer_self_k_cache_updated, n_layer_self_v_cache_updated

class AudioEncoder_KvCache_Base(nn.Module):
    def __init__(self, in_audioEncoder: AudioEncoder_KvCache):
        super().__init__()
        self.audioEncoder = in_audioEncoder

    def forward(self, 
                x: Tensor, 
                self_k_cache0: Tensor, 
                self_k_cache1: Tensor, 
                self_k_cache2: Tensor, 
                self_k_cache3: Tensor, 
                self_k_cache4: Tensor, 
                self_k_cache5: Tensor, 
                self_v_cache0: Tensor, 
                self_v_cache1: Tensor, 
                self_v_cache2: Tensor, 
                self_v_cache3: Tensor, 
                self_v_cache4: Tensor, 
                self_v_cache5: Tensor, 
                positions: Tensor):

        self_k_list = [
            self_k_cache0,
            self_k_cache1,
            self_k_cache2,
            self_k_cache3,
            self_k_cache4,
            self_k_cache5,
            ]
        self_v_list = [
            self_v_cache0,
            self_v_cache1,
            self_v_cache2,
            self_v_cache3,
            self_v_cache4,
            self_v_cache5,
            ]

        cross_k_list, cross_v_list, self_k_cache_update_list, self_v_cache_update_list = self.audioEncoder(x, self_k_list, self_v_list, positions)

        return cross_k_list[0],  cross_k_list[1], cross_k_list[2], cross_k_list[3], cross_k_list[4], cross_k_list[5], cross_v_list[0], cross_v_list[1], cross_v_list[2], cross_v_list[3], cross_v_list[4], cross_v_list[5], self_k_cache_update_list[0], self_k_cache_update_list[1], self_k_cache_update_list[2], self_k_cache_update_list[3], self_k_cache_update_list[4], self_k_cache_update_list[5], self_v_cache_update_list[0], self_v_cache_update_list[1], self_v_cache_update_list[2], self_v_cache_update_list[3], self_v_cache_update_list[4], self_v_cache_update_list[5]

class AudioEncoder_KvCache_Small(nn.Module):
    def __init__(self, in_audioEncoder: AudioEncoder_KvCache):
        super().__init__()
        self.audioEncoder = in_audioEncoder

    def forward(self, 
                x: Tensor, 
                self_k_cache0: Tensor,     self_k_cache1: Tensor,    self_k_cache2: Tensor,    self_k_cache3: Tensor,    self_k_cache4: Tensor,    self_k_cache5: Tensor,    self_k_cache6: Tensor,    self_k_cache7: Tensor,    self_k_cache8: Tensor,    self_k_cache9: Tensor,    self_k_cache10: Tensor,    self_k_cache11: Tensor,
                self_v_cache0: Tensor,     self_v_cache1: Tensor,    self_v_cache2: Tensor,    self_v_cache3: Tensor,    self_v_cache4: Tensor,    self_v_cache5: Tensor,    self_v_cache6: Tensor,    self_v_cache7: Tensor,    self_v_cache8: Tensor,    self_v_cache9: Tensor,    self_v_cache10: Tensor,    self_v_cache11: Tensor,
                positions: Tensor):
        self_k_list = [                                                
            self_k_cache0,    self_k_cache1,    self_k_cache2,    self_k_cache3,    self_k_cache4,    self_k_cache5,    self_k_cache6,    self_k_cache7,    self_k_cache8,    self_k_cache9,    self_k_cache10,    self_k_cache11,
        ]                                                
        self_v_list = [                                                
            self_v_cache0,    self_v_cache1,    self_v_cache2,    self_v_cache3,    self_v_cache4,    self_v_cache5,    self_v_cache6,    self_v_cache7,    self_v_cache8,    self_v_cache9,    self_v_cache10,    self_v_cache11,
        ]                    
        
        cross_k_list, cross_v_list, self_k_cache_update_list, self_v_cache_update_list = self.audioEncoder(x, self_k_list, self_v_list, positions)

        return (                                                
            cross_k_list[0],    cross_k_list[1],    cross_k_list[2],    cross_k_list[3],    cross_k_list[4],    cross_k_list[5],    cross_k_list[6],    cross_k_list[7],    cross_k_list[8],    cross_k_list[9],    cross_k_list[10],    cross_k_list[11],
            cross_v_list[0],    cross_v_list[1],    cross_v_list[2],    cross_v_list[3],    cross_v_list[4],    cross_v_list[5],    cross_v_list[6],    cross_v_list[7],    cross_v_list[8],    cross_v_list[9],    cross_v_list[10],    cross_v_list[11],
            self_k_cache_update_list[0],    self_k_cache_update_list[1],    self_k_cache_update_list[2],    self_k_cache_update_list[3],    self_k_cache_update_list[4],    self_k_cache_update_list[5],    self_k_cache_update_list[6],    self_k_cache_update_list[7],    self_k_cache_update_list[8],    self_k_cache_update_list[9],    self_k_cache_update_list[10],    self_k_cache_update_list[11],
            self_v_cache_update_list[0],    self_v_cache_update_list[1],    self_v_cache_update_list[2],    self_v_cache_update_list[3],    self_v_cache_update_list[4],    self_v_cache_update_list[5],    self_v_cache_update_list[6],    self_v_cache_update_list[7],    self_v_cache_update_list[8],    self_v_cache_update_list[9],    self_v_cache_update_list[10],    self_v_cache_update_list[11],
        )                                                

class AudioEncoder_KvCache_Medium(nn.Module):
    def __init__(self, in_audioEncoder: AudioEncoder_KvCache):
        super().__init__()
        self.audioEncoder = in_audioEncoder

    def forward(self, 
                x: Tensor, 
                self_k_cache0: Tensor,     self_k_cache1: Tensor,    self_k_cache2: Tensor,    self_k_cache3: Tensor,    self_k_cache4: Tensor,    self_k_cache5: Tensor,    self_k_cache6: Tensor,    self_k_cache7: Tensor,    self_k_cache8: Tensor,    self_k_cache9: Tensor,    self_k_cache10: Tensor,    self_k_cache11: Tensor,    self_k_cache12: Tensor,    self_k_cache13: Tensor,    self_k_cache14: Tensor,    self_k_cache15: Tensor,    self_k_cache16: Tensor,    self_k_cache17: Tensor,    self_k_cache18: Tensor,    self_k_cache19: Tensor,    self_k_cache20: Tensor,    self_k_cache21: Tensor,    self_k_cache22: Tensor,    self_k_cache23: Tensor,
                self_v_cache0: Tensor,     self_v_cache1: Tensor,    self_v_cache2: Tensor,    self_v_cache3: Tensor,    self_v_cache4: Tensor,    self_v_cache5: Tensor,    self_v_cache6: Tensor,    self_v_cache7: Tensor,    self_v_cache8: Tensor,    self_v_cache9: Tensor,    self_v_cache10: Tensor,    self_v_cache11: Tensor,    self_v_cache12: Tensor,    self_v_cache13: Tensor,    self_v_cache14: Tensor,    self_v_cache15: Tensor,    self_v_cache16: Tensor,    self_v_cache17: Tensor,    self_v_cache18: Tensor,    self_v_cache19: Tensor,    self_v_cache20: Tensor,    self_v_cache21: Tensor,    self_v_cache22: Tensor,    self_v_cache23: Tensor,
                positions: Tensor):

        self_k_list = [                                                                                                
            self_k_cache0,    self_k_cache1,    self_k_cache2,    self_k_cache3,    self_k_cache4,    self_k_cache5,    self_k_cache6,    self_k_cache7,    self_k_cache8,    self_k_cache9,    self_k_cache10,    self_k_cache11,    self_k_cache12,    self_k_cache13,    self_k_cache14,    self_k_cache15,    self_k_cache16,    self_k_cache17,    self_k_cache18,    self_k_cache19,    self_k_cache20,    self_k_cache21,    self_k_cache22,    self_k_cache23,
        ]                                                                                                
        self_v_list = [                                                                                                
            self_v_cache0,    self_v_cache1,    self_v_cache2,    self_v_cache3,    self_v_cache4,    self_v_cache5,    self_v_cache6,    self_v_cache7,    self_v_cache8,    self_v_cache9,    self_v_cache10,    self_v_cache11,    self_v_cache12,    self_v_cache13,    self_v_cache14,    self_v_cache15,    self_v_cache16,    self_v_cache17,    self_v_cache18,    self_v_cache19,    self_v_cache20,    self_v_cache21,    self_v_cache22,    self_v_cache23,
        ]                                                                                                
        
        cross_k_list, cross_v_list, self_k_cache_update_list, self_v_cache_update_list = self.audioEncoder(x, self_k_list, self_v_list, positions)

        return (                                                                                                
            cross_k_list[0],    cross_k_list[1],    cross_k_list[2],    cross_k_list[3],    cross_k_list[4],    cross_k_list[5],    cross_k_list[6],    cross_k_list[7],    cross_k_list[8],    cross_k_list[9],    cross_k_list[10],    cross_k_list[11],    cross_k_list[12],    cross_k_list[13],    cross_k_list[14],    cross_k_list[15],    cross_k_list[16],    cross_k_list[17],    cross_k_list[18],    cross_k_list[19],    cross_k_list[20],    cross_k_list[21],    cross_k_list[22],    cross_k_list[23],
            cross_v_list[0],    cross_v_list[1],    cross_v_list[2],    cross_v_list[3],    cross_v_list[4],    cross_v_list[5],    cross_v_list[6],    cross_v_list[7],    cross_v_list[8],    cross_v_list[9],    cross_v_list[10],    cross_v_list[11],    cross_v_list[12],    cross_v_list[13],    cross_v_list[14],    cross_v_list[15],    cross_v_list[16],    cross_v_list[17],    cross_v_list[18],    cross_v_list[19],    cross_v_list[20],    cross_v_list[21],    cross_v_list[22],    cross_v_list[23],
            self_k_cache_update_list[0],    self_k_cache_update_list[1],    self_k_cache_update_list[2],    self_k_cache_update_list[3],    self_k_cache_update_list[4],    self_k_cache_update_list[5],    self_k_cache_update_list[6],    self_k_cache_update_list[7],    self_k_cache_update_list[8],    self_k_cache_update_list[9],    self_k_cache_update_list[10],    self_k_cache_update_list[11],    self_k_cache_update_list[12],    self_k_cache_update_list[13],    self_k_cache_update_list[14],    self_k_cache_update_list[15],    self_k_cache_update_list[16],    self_k_cache_update_list[17],    self_k_cache_update_list[18],    self_k_cache_update_list[19],    self_k_cache_update_list[20],    self_k_cache_update_list[21],    self_k_cache_update_list[22],    self_k_cache_update_list[23],
            self_v_cache_update_list[0],    self_v_cache_update_list[1],    self_v_cache_update_list[2],    self_v_cache_update_list[3],    self_v_cache_update_list[4],    self_v_cache_update_list[5],    self_v_cache_update_list[6],    self_v_cache_update_list[7],    self_v_cache_update_list[8],    self_v_cache_update_list[9],    self_v_cache_update_list[10],    self_v_cache_update_list[11],    self_v_cache_update_list[12],    self_v_cache_update_list[13],    self_v_cache_update_list[14],    self_v_cache_update_list[15],    self_v_cache_update_list[16],    self_v_cache_update_list[17],    self_v_cache_update_list[18],    self_v_cache_update_list[19],    self_v_cache_update_list[20],    self_v_cache_update_list[21],    self_v_cache_update_list[22],    self_v_cache_update_list[23],
        )                                                                                                

class TextDecoder_KvCache(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, in_n_ctx: int, in_n_ctx_cache: int, cacheReturnRule: int):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.n_ctx_cache = in_n_ctx_cache #値どこでも使ってないぞい
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block, cacheReturnRule))

    def forward(self, x: Tensor, 
                n_layer_self_k_cache: Tensor, 
                n_layer_self_v_cache: Tensor,
                n_layer_cross_k: Tensor, 
                n_layer_cross_v: Tensor, 
                #offset: Optional[Tensor] = 0
                positions: Optional[Tensor] = None,
                mask: Optional[Tensor] = None
                ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """

        if mask is None:
            mask = self.textDecoder.mask[:x.shape[1], :x.shape[1]]
        #else:
        #    mask *= inf

        if positions is None:
            pos_emb_slice = self.textDecoder.positional_embedding[0:x.shape[1]] #diff!
        else:
            pos_emb_slice = self.textDecoder.positional_embedding[positions] #diff!

        x = self.textDecoder.token_embedding(x) + pos_emb_slice #same
        x = x.to(n_layer_cross_k[0].dtype) #same

        # calc self attention while inputing and outputing kv_cache
        i = 0
        self_k_cache_update_list = []
        self_v_cache_update_list = []
        for block in self.blocks:
            x, self_k_cache_updated, self_v_cache_updated = block(x, 
                                                                  self_k_cache = n_layer_self_k_cache[i], 
                                                                  self_v_cache = n_layer_self_v_cache[i],
                                                                  cross_k = n_layer_cross_k[i], 
                                                                  cross_v = n_layer_cross_v[i], 
                                                                  mask=mask) # diff!
            self_k_cache_update_list.append(self_k_cache_updated)
            self_v_cache_update_list.append(self_v_cache_updated)
            i += 1

        n_layer_self_k_cache_updated = torch.stack(self_k_cache_update_list)
        n_layer_self_v_cache_updated = torch.stack(self_v_cache_update_list)

        #global FOR_ONNX_EXPORT
        if FOR_ONNX_EXPORT:
            x = x[:,-1,:]
            x = self.textDecoder.ln(x) #same
            logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same
            probs = F.softmax(logits, dim=-1)
            return probs, self_k_cache_update_list, self_v_cache_update_list
            #return probs, self_k_cache_update_list, self_v_cache_update_list, logits
        else:
            x = self.textDecoder.ln(x) #same
            logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same
            return logits, n_layer_self_k_cache_updated, n_layer_self_v_cache_updated

class TextDecoder_KvCache_Base(nn.Module):
    def __init__(self, in_TextDecoder_KvCache: TextDecoder_KvCache):
        super().__init__()
        self.textDecoder = in_TextDecoder_KvCache

    def forward(self, 
                x: Tensor, 
                cros_k_cache0: Tensor, 
                cros_k_cache1: Tensor, 
                cros_k_cache2: Tensor, 
                cros_k_cache3: Tensor, 
                cros_k_cache4: Tensor, 
                cros_k_cache5: Tensor, 
                cros_v_cache0: Tensor, 
                cros_v_cache1: Tensor, 
                cros_v_cache2: Tensor, 
                cros_v_cache3: Tensor, 
                cros_v_cache4: Tensor, 
                cros_v_cache5: Tensor, 
                self_k_cache0: Optional[Tensor] = None, 
                self_k_cache1: Optional[Tensor] = None, 
                self_k_cache2: Optional[Tensor] = None, 
                self_k_cache3: Optional[Tensor] = None, 
                self_k_cache4: Optional[Tensor] = None, 
                self_k_cache5: Optional[Tensor] = None, 
                self_v_cache0: Optional[Tensor] = None, 
                self_v_cache1: Optional[Tensor] = None, 
                self_v_cache2: Optional[Tensor] = None, 
                self_v_cache3: Optional[Tensor] = None, 
                self_v_cache4: Optional[Tensor] = None, 
                self_v_cache5: Optional[Tensor] = None, 
                positions: Optional[Tensor] = None,
                mask: Optional[Tensor] = None
                ):

        self_k_list = [
            self_k_cache0,
            self_k_cache1,
            self_k_cache2,
            self_k_cache3,
            self_k_cache4,
            self_k_cache5,
            ]
        self_v_list = [
            self_v_cache0,
            self_v_cache1,
            self_v_cache2,
            self_v_cache3,
            self_v_cache4,
            self_v_cache5,
            ]
        cros_k_list = [
            cros_k_cache0,
            cros_k_cache1,
            cros_k_cache2,
            cros_k_cache3,
            cros_k_cache4,
            cros_k_cache5,
            ]
        cros_v_list = [
            cros_v_cache0,
            cros_v_cache1,
            cros_v_cache2,
            cros_v_cache3,
            cros_v_cache4,
            cros_v_cache5,
            ]

        probs, self_k_cache_update_list, self_v_cache_update_list = self.textDecoder(x, self_k_list, self_v_list, cros_k_list, cros_v_list, positions, mask)

        if self_k_cache0 is None:
            return probs
        else:
            return probs, self_k_cache_update_list[0], self_k_cache_update_list[1], self_k_cache_update_list[2], self_k_cache_update_list[3], self_k_cache_update_list[4], self_k_cache_update_list[5], self_v_cache_update_list[0], self_v_cache_update_list[1], self_v_cache_update_list[2], self_v_cache_update_list[3], self_v_cache_update_list[4], self_v_cache_update_list[5]

        #probs, self_k_cache_update_list, self_v_cache_update_list, logits = self.textDecoder(x, self_k_list, self_v_list, cros_k_list, cros_v_list, positions, mask)
        #return probs, self_k_cache_update_list[0], self_k_cache_update_list[1], self_k_cache_update_list[2], self_k_cache_update_list[3], self_k_cache_update_list[4], self_k_cache_update_list[5], self_v_cache_update_list[0], self_v_cache_update_list[1], self_v_cache_update_list[2], self_v_cache_update_list[3], self_v_cache_update_list[4], self_v_cache_update_list[5], logits

class TextDecoder_KvCache_Small(nn.Module):
    def __init__(self, in_TextDecoder_KvCache: TextDecoder_KvCache):
        super().__init__()
        self.textDecoder = in_TextDecoder_KvCache

    def forward(self, 
                x: Tensor, 
                cros_k_cache0: Tensor,     cros_k_cache1: Tensor,    cros_k_cache2: Tensor,    cros_k_cache3: Tensor,    cros_k_cache4: Tensor,    cros_k_cache5: Tensor,    cros_k_cache6: Tensor,    cros_k_cache7: Tensor,    cros_k_cache8: Tensor,    cros_k_cache9: Tensor,    cros_k_cache10: Tensor,    cros_k_cache11: Tensor,
                cros_v_cache0: Tensor,     cros_v_cache1: Tensor,    cros_v_cache2: Tensor,    cros_v_cache3: Tensor,    cros_v_cache4: Tensor,    cros_v_cache5: Tensor,    cros_v_cache6: Tensor,    cros_v_cache7: Tensor,    cros_v_cache8: Tensor,    cros_v_cache9: Tensor,    cros_v_cache10: Tensor,    cros_v_cache11: Tensor,
                self_k_cache0: Optional[Tensor] = None,     self_k_cache1: Optional[Tensor] = None,    self_k_cache2: Optional[Tensor] = None,    self_k_cache3: Optional[Tensor] = None,    self_k_cache4: Optional[Tensor] = None,    self_k_cache5: Optional[Tensor] = None,    self_k_cache6: Optional[Tensor] = None,    self_k_cache7: Optional[Tensor] = None,    self_k_cache8: Optional[Tensor] = None,    self_k_cache9: Optional[Tensor] = None,    self_k_cache10: Optional[Tensor] = None,    self_k_cache11: Optional[Tensor] = None,
                self_v_cache0: Optional[Tensor] = None,     self_v_cache1: Optional[Tensor] = None,    self_v_cache2: Optional[Tensor] = None,    self_v_cache3: Optional[Tensor] = None,    self_v_cache4: Optional[Tensor] = None,    self_v_cache5: Optional[Tensor] = None,    self_v_cache6: Optional[Tensor] = None,    self_v_cache7: Optional[Tensor] = None,    self_v_cache8: Optional[Tensor] = None,    self_v_cache9: Optional[Tensor] = None,    self_v_cache10: Optional[Tensor] = None,    self_v_cache11: Optional[Tensor] = None,
                positions: Optional[Tensor] = None,
                mask: Optional[Tensor] = None
                ):

        self_k_list = [                                                
            self_k_cache0,    self_k_cache1,    self_k_cache2,    self_k_cache3,    self_k_cache4,    self_k_cache5,    self_k_cache6,    self_k_cache7,    self_k_cache8,    self_k_cache9,    self_k_cache10,    self_k_cache11,
        ]                                                
        self_v_list = [                                                
            self_v_cache0,    self_v_cache1,    self_v_cache2,    self_v_cache3,    self_v_cache4,    self_v_cache5,    self_v_cache6,    self_v_cache7,    self_v_cache8,    self_v_cache9,    self_v_cache10,    self_v_cache11,
        ]                                                
        cros_k_list = [                                                
            cros_k_cache0,    cros_k_cache1,    cros_k_cache2,    cros_k_cache3,    cros_k_cache4,    cros_k_cache5,    cros_k_cache6,    cros_k_cache7,    cros_k_cache8,    cros_k_cache9,    cros_k_cache10,    cros_k_cache11,
        ]                                                
        cros_v_list = [                                                
            cros_v_cache0,    cros_v_cache1,    cros_v_cache2,    cros_v_cache3,    cros_v_cache4,    cros_v_cache5,    cros_v_cache6,    cros_v_cache7,    cros_v_cache8,    cros_v_cache9,    cros_v_cache10,    cros_v_cache11,
        ]

        probs, self_k_cache_update_list, self_v_cache_update_list = self.textDecoder(x, self_k_list, self_v_list, cros_k_list, cros_v_list, positions, mask)
    
        if self_k_cache0 is None:
            return probs
        else:
            return (probs,
                self_k_cache_update_list[0],    self_k_cache_update_list[1],    self_k_cache_update_list[2],    self_k_cache_update_list[3],    self_k_cache_update_list[4],    self_k_cache_update_list[5],    self_k_cache_update_list[6],    self_k_cache_update_list[7],    self_k_cache_update_list[8],    self_k_cache_update_list[9],    self_k_cache_update_list[10],    self_k_cache_update_list[11],
                self_v_cache_update_list[0],    self_v_cache_update_list[1],    self_v_cache_update_list[2],    self_v_cache_update_list[3],    self_v_cache_update_list[4],    self_v_cache_update_list[5],    self_v_cache_update_list[6],    self_v_cache_update_list[7],    self_v_cache_update_list[8],    self_v_cache_update_list[9],    self_v_cache_update_list[10],    self_v_cache_update_list[11],
            )

class TextDecoder_KvCache_Medium(nn.Module):
    def __init__(self, in_TextDecoder_KvCache: TextDecoder_KvCache):
        super().__init__()
        self.textDecoder = in_TextDecoder_KvCache

    def forward(self, 
                x: Tensor, 
                cros_k_cache0: Tensor,     cros_k_cache1: Tensor,    cros_k_cache2: Tensor,    cros_k_cache3: Tensor,    cros_k_cache4: Tensor,    cros_k_cache5: Tensor,    cros_k_cache6: Tensor,    cros_k_cache7: Tensor,    cros_k_cache8: Tensor,    cros_k_cache9: Tensor,    cros_k_cache10: Tensor,    cros_k_cache11: Tensor,    cros_k_cache12: Tensor,    cros_k_cache13: Tensor,    cros_k_cache14: Tensor,    cros_k_cache15: Tensor,    cros_k_cache16: Tensor,    cros_k_cache17: Tensor,    cros_k_cache18: Tensor,    cros_k_cache19: Tensor,    cros_k_cache20: Tensor,    cros_k_cache21: Tensor,    cros_k_cache22: Tensor,    cros_k_cache23: Tensor,
                cros_v_cache0: Tensor,     cros_v_cache1: Tensor,    cros_v_cache2: Tensor,    cros_v_cache3: Tensor,    cros_v_cache4: Tensor,    cros_v_cache5: Tensor,    cros_v_cache6: Tensor,    cros_v_cache7: Tensor,    cros_v_cache8: Tensor,    cros_v_cache9: Tensor,    cros_v_cache10: Tensor,    cros_v_cache11: Tensor,    cros_v_cache12: Tensor,    cros_v_cache13: Tensor,    cros_v_cache14: Tensor,    cros_v_cache15: Tensor,    cros_v_cache16: Tensor,    cros_v_cache17: Tensor,    cros_v_cache18: Tensor,    cros_v_cache19: Tensor,    cros_v_cache20: Tensor,    cros_v_cache21: Tensor,    cros_v_cache22: Tensor,    cros_v_cache23: Tensor,
                self_k_cache0: Optional[Tensor] = None,     self_k_cache1: Optional[Tensor] = None,    self_k_cache2: Optional[Tensor] = None,    self_k_cache3: Optional[Tensor] = None,    self_k_cache4: Optional[Tensor] = None,    self_k_cache5: Optional[Tensor] = None,    self_k_cache6: Optional[Tensor] = None,    self_k_cache7: Optional[Tensor] = None,    self_k_cache8: Optional[Tensor] = None,    self_k_cache9: Optional[Tensor] = None,    self_k_cache10: Optional[Tensor] = None,    self_k_cache11: Optional[Tensor] = None,    self_k_cache12: Optional[Tensor] = None,    self_k_cache13: Optional[Tensor] = None,    self_k_cache14: Optional[Tensor] = None,    self_k_cache15: Optional[Tensor] = None,    self_k_cache16: Optional[Tensor] = None,    self_k_cache17: Optional[Tensor] = None,    self_k_cache18: Optional[Tensor] = None,    self_k_cache19: Optional[Tensor] = None,    self_k_cache20: Optional[Tensor] = None,    self_k_cache21: Optional[Tensor] = None,    self_k_cache22: Optional[Tensor] = None,    self_k_cache23: Optional[Tensor] = None,
                self_v_cache0: Optional[Tensor] = None,     self_v_cache1: Optional[Tensor] = None,    self_v_cache2: Optional[Tensor] = None,    self_v_cache3: Optional[Tensor] = None,    self_v_cache4: Optional[Tensor] = None,    self_v_cache5: Optional[Tensor] = None,    self_v_cache6: Optional[Tensor] = None,    self_v_cache7: Optional[Tensor] = None,    self_v_cache8: Optional[Tensor] = None,    self_v_cache9: Optional[Tensor] = None,    self_v_cache10: Optional[Tensor] = None,    self_v_cache11: Optional[Tensor] = None,    self_v_cache12: Optional[Tensor] = None,    self_v_cache13: Optional[Tensor] = None,    self_v_cache14: Optional[Tensor] = None,    self_v_cache15: Optional[Tensor] = None,    self_v_cache16: Optional[Tensor] = None,    self_v_cache17: Optional[Tensor] = None,    self_v_cache18: Optional[Tensor] = None,    self_v_cache19: Optional[Tensor] = None,    self_v_cache20: Optional[Tensor] = None,    self_v_cache21: Optional[Tensor] = None,    self_v_cache22: Optional[Tensor] = None,    self_v_cache23: Optional[Tensor] = None,
                positions: Optional[Tensor] = None,
                mask: Optional[Tensor] = None
                ):

        self_k_list = [                                                                                                
            self_k_cache0,    self_k_cache1,    self_k_cache2,    self_k_cache3,    self_k_cache4,    self_k_cache5,    self_k_cache6,    self_k_cache7,    self_k_cache8,    self_k_cache9,    self_k_cache10,    self_k_cache11,    self_k_cache12,    self_k_cache13,    self_k_cache14,    self_k_cache15,    self_k_cache16,    self_k_cache17,    self_k_cache18,    self_k_cache19,    self_k_cache20,    self_k_cache21,    self_k_cache22,    self_k_cache23,
        ]                                                                                                
        self_v_list = [                                                                                                
            self_v_cache0,    self_v_cache1,    self_v_cache2,    self_v_cache3,    self_v_cache4,    self_v_cache5,    self_v_cache6,    self_v_cache7,    self_v_cache8,    self_v_cache9,    self_v_cache10,    self_v_cache11,    self_v_cache12,    self_v_cache13,    self_v_cache14,    self_v_cache15,    self_v_cache16,    self_v_cache17,    self_v_cache18,    self_v_cache19,    self_v_cache20,    self_v_cache21,    self_v_cache22,    self_v_cache23,
        ]                                                                                                
        cros_k_list = [                                                                                                
            cros_k_cache0,    cros_k_cache1,    cros_k_cache2,    cros_k_cache3,    cros_k_cache4,    cros_k_cache5,    cros_k_cache6,    cros_k_cache7,    cros_k_cache8,    cros_k_cache9,    cros_k_cache10,    cros_k_cache11,    cros_k_cache12,    cros_k_cache13,    cros_k_cache14,    cros_k_cache15,    cros_k_cache16,    cros_k_cache17,    cros_k_cache18,    cros_k_cache19,    cros_k_cache20,    cros_k_cache21,    cros_k_cache22,    cros_k_cache23,
        ]                                                                                                
        cros_v_list = [                                                                                                
            cros_v_cache0,    cros_v_cache1,    cros_v_cache2,    cros_v_cache3,    cros_v_cache4,    cros_v_cache5,    cros_v_cache6,    cros_v_cache7,    cros_v_cache8,    cros_v_cache9,    cros_v_cache10,    cros_v_cache11,    cros_v_cache12,    cros_v_cache13,    cros_v_cache14,    cros_v_cache15,    cros_v_cache16,    cros_v_cache17,    cros_v_cache18,    cros_v_cache19,    cros_v_cache20,    cros_v_cache21,    cros_v_cache22,    cros_v_cache23,
        ]                                                                                                

        probs, self_k_cache_update_list, self_v_cache_update_list = self.textDecoder(x, self_k_list, self_v_list, cros_k_list, cros_v_list, positions, mask)
    
        if self_k_cache0 is None:
            return probs
        else:
            return (probs,                                                                                                
                self_k_cache_update_list[0],    self_k_cache_update_list[1],    self_k_cache_update_list[2],    self_k_cache_update_list[3],    self_k_cache_update_list[4],    self_k_cache_update_list[5],    self_k_cache_update_list[6],    self_k_cache_update_list[7],    self_k_cache_update_list[8],    self_k_cache_update_list[9],    self_k_cache_update_list[10],    self_k_cache_update_list[11],    self_k_cache_update_list[12],    self_k_cache_update_list[13],    self_k_cache_update_list[14],    self_k_cache_update_list[15],    self_k_cache_update_list[16],    self_k_cache_update_list[17],    self_k_cache_update_list[18],    self_k_cache_update_list[19],    self_k_cache_update_list[20],    self_k_cache_update_list[21],    self_k_cache_update_list[22],    self_k_cache_update_list[23],
                self_v_cache_update_list[0],    self_v_cache_update_list[1],    self_v_cache_update_list[2],    self_v_cache_update_list[3],    self_v_cache_update_list[4],    self_v_cache_update_list[5],    self_v_cache_update_list[6],    self_v_cache_update_list[7],    self_v_cache_update_list[8],    self_v_cache_update_list[9],    self_v_cache_update_list[10],    self_v_cache_update_list[11],    self_v_cache_update_list[12],    self_v_cache_update_list[13],    self_v_cache_update_list[14],    self_v_cache_update_list[15],    self_v_cache_update_list[16],    self_v_cache_update_list[17],    self_v_cache_update_list[18],    self_v_cache_update_list[19],    self_v_cache_update_list[20],    self_v_cache_update_list[21],    self_v_cache_update_list[22],    self_v_cache_update_list[23],
            )                                                                                                

class TextDecoder_KvCache_NoSelfCache(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder_KvCache, n_layer: int):
        super().__init__()
        self.textDecoder_KvCache = in_textDecoder

        self.NoneList = []
        for i in range(n_layer):
            self.NoneList.append(None)

    def forward(self, x: Tensor, 
                n_layer_cross_k: Tensor, 
                n_layer_cross_v: Tensor, 
                #offset: Optional[Tensor] = 0
                positions: Optional[Tensor] = None
                ):

        x,_,_ = self.textDecoder_KvCache(x, self.NoneList, self.NoneList, n_layer_cross_k, n_layer_cross_v, positions)
        return x

class WhisperPreKV(nn.Module):
    def __init__(self, in_whisper: Whisper):
        super().__init__()
        self.whisper = in_whisper
        self.dims = self.whisper.dims
        self.encoder = AudioEncoder_KvCache(self.whisper.encoder, self.whisper.decoder, 1500, 0) #TODO まずはキャッシュやめて動かす
        self.decoder = TextDecoder_KvCache(self.whisper.decoder, 0, 0, 0)  #TODO まずは動的にn_ctx変えて動かす
        #self.decoder16tkn = TextDecoderPreKv16tkn(self.whisper.decoder)
        #self.decoder1tkn = TextDecoderPreKv1tkn(self.whisper.decoder)

    def logits(self, tokens: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.decoder.forward(tokens, k, v, 0)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        return self.whisper.install_kv_cache_hooks(cache)

    @property
    def device(self):
        return self.whisper.device

    @property
    def is_multilingual(self):
        return self.whisper.is_multilingual

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

    def exportOnnxEncoder(self, name, n_ctx: int, n_ctx_cache: int, isDynamicIn: bool, isDynamicCacheIn: bool):
        device = self.whisper.device
        n_layer = self.dims.n_text_layer
        n_state = self.dims.n_text_state
        #n_ctx_cache = 1500 - n_ctx
        n_mel = n_ctx * 2
        offset = 0
        self.encoder.n_ctx = n_ctx

        dummy_mel =     torch.randn((1, n_mel, 80), dtype=torch.float32).to(device)
        dummy_k_cache = torch.randn((n_layer, 1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        dummy_v_cache = torch.randn((n_layer, 1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        #dummy_offset =  torch.tensor(offset, dtype=torch.int64).to(device).unsqueeze(0)
        dummy_positions = torch.arange(offset, offset+n_ctx, 1).to(device)

        #inputs = ( dummy_mel, dummy_k_cache, dummy_v_cache, dummy_offset )
        inputs = ( dummy_mel, dummy_k_cache, dummy_v_cache, dummy_positions )
        input_names = ['mel_t', 'self_k_in', 'self_v_in', 'positions']
        output_names = ['cross_k', 'cross_v', 'self_k_out', 'self_v_out']

        file_base = "encoder_norm_"
        dynamic_axes = dict()

        if isDynamicIn:
            file_base += "-1_"
            dynamic_axes['mel_t'] = {1: 'n_mel'}
            dynamic_axes['positions'] = {0: 'n_ctx'}
        else:
            file_base += str(n_ctx) + "_"

        if isDynamicCacheIn:
            file_base += "-1_"
            dynamic_axes['self_k_in'] = {2: 'n_ctx_cache'}
            dynamic_axes['self_v_in'] = {2: 'n_ctx_cache'}
        else:
            file_base += str(n_ctx_cache) + "_"

        file_base += name
        file_onnx = file_base + ".onnx"
        file_simp = file_base + "_smpl.onnx"

        torch.onnx.export(self.encoder,
                        inputs,
                        file_onnx,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
        )
        onnx_model = onnx.load(f'{file_onnx}')
        onnx_model_simp, check = simplify(onnx_model)
        onnx.save(onnx_model_simp, f'{file_simp}')

    def exportOnnxEncoder_EachLayer(self, name, n_ctx: int, n_ctx_cache: int, isDynamicIn: bool, isDynamicCacheIn: bool):
        device = self.whisper.device
        n_layer = self.dims.n_text_layer
        n_state = self.dims.n_text_state
        #n_ctx_cache = 1500 - n_ctx
        n_mel = n_ctx * 2
        offset = 0
        self.encoder.n_ctx = n_ctx

        dummy_mel =     torch.randn((1, n_mel, 80), dtype=torch.float32).to(device)
        dummy_k_cache = torch.randn((1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        dummy_v_cache = torch.randn((1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        #dummy_offset =  torch.tensor(offset, dtype=torch.int64).to(device).unsqueeze(0)
        dummy_positions = torch.arange(offset, offset+n_ctx, 1).to(device)

        if name == 'base' or name == 'base.en':
            self.encoder_EL = AudioEncoder_KvCache_Base(self.encoder)
            inputs = ( dummy_mel, dummy_k_cache,dummy_k_cache,dummy_k_cache,dummy_k_cache,dummy_k_cache,dummy_k_cache, dummy_v_cache,dummy_v_cache,dummy_v_cache,dummy_v_cache,dummy_v_cache,dummy_v_cache, dummy_positions )

        if name == 'small' or name == 'small.en':
            self.encoder_EL = AudioEncoder_KvCache_Small(self.encoder)
            inputs = (dummy_mel,                                                
                dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,
                dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,
                dummy_positions)                                                

        if name == 'medium' or name == 'medium.en':
            self.encoder_EL = AudioEncoder_KvCache_Medium(self.encoder)
            inputs = (dummy_mel,                                                                                                
                dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,    dummy_k_cache,
                dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,    dummy_v_cache,
                dummy_positions)                                                                                                

        input_names = ['mel_t']
        for i in range(n_layer):
            input_names.append('self_k_in' + str(i))
        for i in range(n_layer):
            input_names.append('self_v_in' + str(i))
        input_names.append('positions')

        output_names = []
        for i in range(n_layer):
            output_names.append('cross_k' + str(i))
        for i in range(n_layer):
            output_names.append('cross_v' + str(i))
        for i in range(n_layer):
            output_names.append('self_k_out' + str(i))
        for i in range(n_layer):
            output_names.append('self_v_out' + str(i))

        #file_base = "encoder_el_"
        file_base = "encoder_norm_el_"
        dynamic_axes = dict()

        if isDynamicIn:
            file_base += "-1_"
            dynamic_axes['mel_t'] = {1: 'n_mel'}
            dynamic_axes['positions'] = {0: 'n_ctx'}
        else:
            file_base += str(n_ctx) + "_"

        if isDynamicCacheIn:
            file_base += "-1_"
            for i in range(n_layer):
                dynamic_axes['self_k_in' + str(i)] = {1: 'n_ctx_cache'}
            for i in range(n_layer):
                dynamic_axes['self_v_in' + str(i)] = {1: 'n_ctx_cache'}
        else:
            file_base += str(n_ctx_cache) + "_"

        file_base += name
        file_onnx = file_base + ".onnx"
        file_simp = file_base + "_smpl.onnx"

        torch.onnx.export(self.encoder_EL,
                        inputs,
                        file_onnx,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
        )
        onnx_model = onnx.load(f'{file_onnx}')
        onnx_model_simp, check = simplify(onnx_model)
        onnx.save(onnx_model_simp, f'{file_simp}')

    def exportOnnxDecoder(self, name, n_ctx: int, n_ctx_cache: int, isDynamicIn: bool, isDynamicCacheIn: bool):
        if isDynamicIn:
            self.decoderE = TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=0)
        else:
            self.decoderE = TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2)
        if n_ctx_cache == 0:
            self.decoderNoSelfCache = TextDecoder_KvCache_NoSelfCache(self.decoderE, self.dims.n_text_layer)

        device = self.whisper.device
        n_state = self.dims.n_text_state
        n_layer = self.dims.n_text_layer
        offset = 0

        token_list = []
        for i in range(n_ctx):
            token_list.append(torch.tensor(0, dtype=torch.int64).to(device))
        dummy_tokens = torch.stack(token_list).unsqueeze(0)
        dummy_self_k = torch.randn((n_layer, 1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        dummy_self_v = torch.randn((n_layer, 1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        dummy_cross_k = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to(device) 
        dummy_cross_v = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to(device)
        dummy_offset = torch.tensor(offset, dtype=torch.int64).to(device)
        dummy_positions = torch.arange(offset, offset+n_ctx, 1).to(device)
        dummy_mask = torch.ones(n_ctx, n_ctx).to(device)

        if n_ctx_cache == 0:
            if isDynamicIn:
                # no more needed!
                inputs = ( dummy_tokens, dummy_cross_k, dummy_cross_v, dummy_positions )
                input_names = ['tokens', 'cross_k', 'cross_v', 'positions']
            else:
                inputs = ( dummy_tokens, dummy_cross_k, dummy_cross_v )
                input_names = ['tokens', 'cross_k', 'cross_v']
            output_names = ['probabilities']
            decoder = self.decoderNoSelfCache
        else:
            if isDynamicIn:
                inputs = ( dummy_tokens, dummy_self_k, dummy_self_v, dummy_cross_k, dummy_cross_v, dummy_positions, dummy_mask )
                input_names = ['tokens', 'self_k_in', 'self_v_in', 'cross_k', 'cross_v', 'positions', 'mask']
                output_names = ['probabilities', 'self_k_out', 'self_v_out']
            else:
                inputs = ( dummy_tokens, dummy_self_k, dummy_self_v, dummy_cross_k, dummy_cross_v, dummy_positions )
                input_names = ['tokens', 'self_k_in', 'self_v_in', 'cross_k', 'cross_v', 'positions']
                output_names = ['probabilities', 'self_k_out', 'self_v_out']
            decoder = self.decoderE

        file_base = "decoder_"
        dynamic_axes = dict()

        if isDynamicIn:
            file_base += "-1_"
            dynamic_axes['tokens'] = {1: 'n_ctx'}
            dynamic_axes['positions'] = {0: 'n_ctx'}
            if n_ctx_cache > 0:
               dynamic_axes['mask'] = {0: 'n_ctx', 1: 'n_ctx'}
        else:
            file_base += str(n_ctx) + "_"

        if isDynamicCacheIn:
            file_base += "-1_"
            dynamic_axes['self_k_in'] = {2: 'n_ctx_cache'}
            dynamic_axes['self_v_in'] = {2: 'n_ctx_cache'}
        else:
            file_base += str(n_ctx_cache) + "_"

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
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
        )
        onnx_model = onnx.load(f'{file_onnx}')
        onnx_model_simp, check = simplify(onnx_model)
        onnx.save(onnx_model_simp, f'{file_simp}')

    def exportOnnxDecoder_EachLayer(self, name, n_ctx: int, n_ctx_cache: int, isDynamicIn: bool, isDynamicCacheIn: bool):
        device = self.whisper.device
        n_state = self.dims.n_text_state
        n_layer = self.dims.n_text_layer
        offset = 0

        token_list = []
        for i in range(n_ctx):
            token_list.append(torch.tensor(0, dtype=torch.int64).to(device))
        dummy_tokens = torch.stack(token_list).unsqueeze(0)
        dummy_self_k = torch.randn((1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        dummy_self_v = torch.randn((1, n_ctx_cache, n_state), dtype=torch.float32).to(device)
        dummy_cros_k = torch.randn((1, 1500, n_state), dtype=torch.float32).to(device) 
        dummy_cros_v = torch.randn((1, 1500, n_state), dtype=torch.float32).to(device)
        dummy_offset = torch.tensor(offset, dtype=torch.int64).to(device)
        dummy_positions = torch.arange(offset, offset+n_ctx, 1).to(device)
        dummy_mask = torch.ones(n_ctx, n_ctx).to(device)

        if name == 'base' or name == 'base.en':
            decoder = TextDecoder_KvCache_Base(
                TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2) #return only new cache (n_ctx)
                )
            inputs = ( dummy_tokens, 
                      dummy_cros_k,dummy_cros_k,dummy_cros_k,dummy_cros_k,dummy_cros_k,dummy_cros_k, 
                      dummy_cros_v,dummy_cros_v,dummy_cros_v,dummy_cros_v,dummy_cros_v,dummy_cros_v, 
                      dummy_self_k,dummy_self_k,dummy_self_k,dummy_self_k,dummy_self_k,dummy_self_k, 
                      dummy_self_v,dummy_self_v,dummy_self_v,dummy_self_v,dummy_self_v,dummy_self_v, 
                      dummy_positions, dummy_mask )

        if name == 'small' or name == 'small.en':
            decoder = TextDecoder_KvCache_Small(
                TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2) #return only new cache (n_ctx)
                )
            inputs = ( dummy_tokens,                                                 
                dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k, 
                dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v, 
                dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,
                dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,
                dummy_positions, dummy_mask )                                                

        if name == 'medium' or name == 'medium.en':
            decoder = TextDecoder_KvCache_Medium(
                TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2) #return only new cache (n_ctx)
                )
            inputs = ( dummy_tokens,                                                                                                 
                dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k, 
                dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v, 
                dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,    dummy_self_k,
                dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,    dummy_self_v,
                dummy_positions, dummy_mask )                                                                                                

        input_names = ['tokens']
        for i in range(n_layer):
            input_names.append('cross_k' + str(i))
        for i in range(n_layer):
            input_names.append('cross_v' + str(i))
        for i in range(n_layer):
            input_names.append('self_k_in' + str(i))
        for i in range(n_layer):
            input_names.append('self_v_in' + str(i))
        input_names.append('positions')
        input_names.append('mask')

        output_names = ['probabilities']
        for i in range(n_layer):
            output_names.append('self_k_out' + str(i))
        for i in range(n_layer):
            output_names.append('self_v_out' + str(i))
        #output_names.append('logits')

        # From el2, cross kv input is before self kv input
        file_base = "decoder_el2_"
        #file_base = "decoder_el_lp_"
        dynamic_axes = dict()

        if isDynamicIn:
            file_base += "-1_"
            dynamic_axes['tokens'] = {1: 'n_ctx'}
            dynamic_axes['positions'] = {0: 'n_ctx'}
            if n_ctx_cache > 0:
               dynamic_axes['mask'] = {0: 'n_ctx', 1: 'n_ctx'}
        else:
            file_base += str(n_ctx) + "_"

        if isDynamicCacheIn:
            file_base += "-1_"
            for i in range(n_layer):
                dynamic_axes['self_k_in' + str(i)] = {1: 'n_ctx_cache'}
            for i in range(n_layer):
                dynamic_axes['self_v_in' + str(i)] = {1: 'n_ctx_cache'}

            #file_base += "-1_"
            #for i in range(n_layer):
            #    dynamic_axes['cross_k' + str(i)] = {1: 'n_ctx_cache'}
            #for i in range(n_layer):
            #    dynamic_axes['cross_v' + str(i)] = {1: 'n_ctx_cache'}
        else:
            file_base += str(n_ctx_cache) + "_"

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
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
        )
        onnx_model = onnx.load(f'{file_onnx}')
        onnx_model_simp, check = simplify(onnx_model)
        onnx.save(onnx_model_simp, f'{file_simp}')

    def exportOnnxDecoder_EachLayer_Static_NoSelfCache(self, name, n_ctx: int):
        n_ctx_cache = 0
        device = self.whisper.device
        n_state = self.dims.n_text_state
        n_layer = self.dims.n_text_layer
        offset = 0

        token_list = []
        for i in range(n_ctx):
            token_list.append(torch.tensor(0, dtype=torch.int64).to(device))
        dummy_tokens = torch.stack(token_list).unsqueeze(0)
        dummy_cros_k = torch.randn((1, 1500, n_state), dtype=torch.float32).to(device) 
        dummy_cros_v = torch.randn((1, 1500, n_state), dtype=torch.float32).to(device)
        dummy_offset = torch.tensor(offset, dtype=torch.int64).to(device)
        dummy_positions = torch.arange(offset, offset+n_ctx, 1).to(device)
        dummy_mask = torch.ones(n_ctx, n_ctx).to(device)

        if name == 'base' or name == 'base.en':
            decoder = TextDecoder_KvCache_Base(
                TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2) #return only new cache (n_ctx)
                )
            inputs = ( dummy_tokens, 
                      dummy_cros_k,dummy_cros_k,dummy_cros_k,dummy_cros_k,dummy_cros_k,dummy_cros_k, 
                      dummy_cros_v,dummy_cros_v,dummy_cros_v,dummy_cros_v,dummy_cros_v,dummy_cros_v, 
                      )

        if name == 'small' or name == 'small.en':
            decoder = TextDecoder_KvCache_Small(
                TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2) #return only new cache (n_ctx)
                )
            inputs = ( dummy_tokens,                                                 
                dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k, 
                dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v, 
                )

        if name == 'medium' or name == 'medium.en':
            decoder = TextDecoder_KvCache_Medium(
                TextDecoder_KvCache(self.whisper.decoder, n_ctx, n_ctx_cache, cacheReturnRule=2) #return only new cache (n_ctx)
                )
            inputs = ( dummy_tokens,                                                                                                 
                dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k,     dummy_cros_k, 
                dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v,     dummy_cros_v, 
                )

        input_names = ['tokens']
        for i in range(n_layer):
            input_names.append('cross_k' + str(i))
        for i in range(n_layer):
            input_names.append('cross_v' + str(i))

        output_names = ['probabilities']
        #output_names.append('logits')

        file_base = "decoder_el_"
        #file_base = "decoder_el_lp_"

        file_base += str(n_ctx) + "_"
        file_base += str(n_ctx_cache) + "_"

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

    def exportOnnxEncoder_orginal(self, name, n_ctx: int):
        device = self.whisper.device
        n_layer = self.dims.n_text_layer
        n_state = self.dims.n_text_state
        n_mel = n_ctx * 2
        n_ctx_cache = 0

        dummy_mel =     torch.randn((1, n_mel, 80), dtype=torch.float32).to(device)

        inputs = ( dummy_mel )
        input_names = ['mel']
        output_names = ['audio_feature']

        file_base = "encoder_org_"
        file_base += str(n_ctx) + "_"
        file_base += str(n_ctx_cache) + "_"
        file_base += name
        file_onnx = file_base + ".onnx"
        file_simp = file_base + "_smpl.onnx"

        torch.onnx.export(self.whisper.encoder,
                        inputs,
                        file_onnx,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names
        )
        onnx_model = onnx.load(f'{file_onnx}')
        #onnx_model_simp, check = simplify(onnx_model)
        #onnx.save(onnx_model_simp, f'{file_simp}')