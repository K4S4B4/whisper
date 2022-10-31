from dataclasses import dataclass
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


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
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
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            #qk = qk + mask[:n_ctx, :n_ctx] #これが出てくるのはSelfAttentionのとき。queryのTokenより未来のTokenのkeyを問い合わせてるときは-Infにしてしまう。いや、これ要るのか？未来情報で過去を改善できそうなのだが。
            n_ctx_cache = k.shape[-1] - n_ctx
            mask0 = torch.zeros((n_ctx, n_ctx_cache), dtype=q.dtype).to(q.device);
            mask_cat = torch.hstack((mask0, mask[:n_ctx, :n_ctx])) #(n_ctx, n_ctx_cache + n_ctx)
            qk = qk + mask_cat

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


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

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        #xnum = x.squeeze().to('cpu').detach().numpy().copy().astype(np.float32)
        #cv2.imshow("EncAfterConv", xnum)
        #cv2.waitKey(0)

        x = x.permute(0, 2, 1)


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

    def forward(
        self,
        x: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        wv = self.multiHeadAttention.qkv_attention(q, cross_k, cross_v, mask)
        return self.multiHeadAttention.out(wv)

class MultiHeadAttention_SelfKvCache(nn.Module):
    def __init__(self, in_multiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = in_multiHeadAttention

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor, #(1, n_ctx_cache, 512)
        self_v_cache: Tensor, #(1, n_ctx_cache, 512)
        mask: Optional[Tensor] = None,
    ):
        n_ctx = x.shape[1]

        q = self.multiHeadAttention.query(x)
        k = self.multiHeadAttention.key(x)   #(1, n_ctx, 512)
        v = self.multiHeadAttention.value(x) #(1, n_ctx, 512)
        self_k_cache_updated = torch.cat((self_k_cache, k), 1) #(1, n_ctx_cache + n_ctx, 512)
        self_v_cache_updated = torch.cat((self_v_cache, v), 1) #(1, n_ctx_cache + n_ctx, 512)
        wv = self.multiHeadAttention.qkv_attention(q, self_k_cache_updated, self_v_cache_updated, mask)

        #TODO enable static shape caching!!!
        #n_ctx_cache = self_k_cache.shape[1]
        #self_k_cache_updated = self_k_cache_updated[:,n_ctx:n_ctx+n_ctx_cache,:] #(1, n_ctx_cache + n_ctx, 512) -> (1, n_ctx_cache, 512). Get last (n_ctx_cache)
        #self_v_cache_updated = self_v_cache_updated[:,n_ctx:n_ctx+n_ctx_cache,:] #(1, n_ctx_cache + n_ctx, 512) -> (1, n_ctx_cache, 512)

        return self.multiHeadAttention.out(wv), self_k_cache_updated, self_v_cache_updated

class ResidualAttentionBlock_KvCache(nn.Module):
    def __init__(self, in_residualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = in_residualAttentionBlock
        self.attn = MultiHeadAttention_SelfKvCache(in_residualAttentionBlock.attn)
        self.cross_attn = MultiHeadAttention_CrossKvCache(in_residualAttentionBlock.cross_attn) if in_residualAttentionBlock.cross_attn else None

    def forward(
        self,
        x: Tensor,
        self_k_cache: Optional[Tensor] = None,
        self_v_cache: Optional[Tensor] = None,
        cross_k: Optional[Tensor] = None,
        cross_v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, mask=mask) #diff!
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
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block))

    def forward(self, x: Tensor, n_layer_self_k_cache: Tensor, n_layer_self_v_cache: Tensor, offset: int):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        n_layer_self_k_cache : shape = (n_layer, 1, n_ctx_cache, 512)
        n_layer_self_v_cache : shape = (n_layer, 1, n_ctx_cache, 512)
        offset : where the positional_embedding starts for x. [0, 1499]
        return : n_layer_cross_k, n_layer_cross_v : precomputed kv for cross attention from decoder's query. shape = (n_layer, 1, n_ctx, 512)
        return : n_layer_self_k_cache_updated, n_layer_self_v_cache_updated : precomputed kv for self attention from encoder's query. shape = (n_layer, 1, n_ctx_cache + n_ctx, 512)
        """

        # pre process
        x = x.permute(0, 2, 1) #diff

        x = F.gelu(self.audioEncoder.conv1(x)) #same
        x = F.gelu(self.audioEncoder.conv2(x)) #same
        x = x.permute(0, 2, 1) #same

        x = (x + self.audioEncoder.positional_embedding[offset:offset+self.n_ctx]).to(x.dtype) #diff

        #diff
        # calc self attention while inputing and outputing kv_cache
        i = 0
        self_k_cache_update_list = []
        self_v_cache_update_list = []
        for block in self.blocks:
            x, self_k_cache_updated, self_v_cache_updated = block(x, 
                                                                  self_k_cache = n_layer_self_k_cache[i], 
                                                                  self_v_cache = n_layer_self_v_cache[i]) 
            self_k_cache_update_list.append(self_k_cache_updated)
            self_v_cache_update_list.append(self_v_cache_updated)
            i += 1

        n_layer_self_k_cache_updated = torch.stack(self_k_cache_update_list)
        n_layer_self_v_cache_updated = torch.stack(self_v_cache_update_list)

        xa = self.audioEncoder.ln_post(x) #same

        # pre compute key-value for audio feature
        cross_k_list = []
        cross_v_list = []
        for block in self.textDecoder.blocks:
            if block.cross_attn:
                cross_k_list.append(block.cross_attn.key(xa))
                cross_v_list.append(block.cross_attn.value(xa))

        n_layer_cross_k = torch.stack(cross_k_list)
        n_layer_cross_v = torch.stack(cross_v_list)

        return n_layer_cross_k, n_layer_cross_v, n_layer_self_k_cache_updated, n_layer_self_v_cache_updated

class TextDecoder_KvCache(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, in_n_ctx: int, in_n_ctx_cache: int):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.n_ctx_cache = in_n_ctx_cache #値どこでも使ってないぞい
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block))

    def forward(self, x: Tensor, 
                n_layer_self_k_cache: Tensor, 
                n_layer_self_v_cache: Tensor,
                n_layer_cross_k: Tensor, 
                n_layer_cross_v: Tensor, 
                offset: Optional[Tensor] = 0
                ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """

        pos_emb_slice = self.textDecoder.positional_embedding[offset : offset + self.n_ctx] #diff!
        x = self.textDecoder.token_embedding(x) + pos_emb_slice #same
        x = x.to(n_layer_cross_k.dtype) #same

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
                                                                  mask=self.textDecoder.mask) # diff!
            self_k_cache_update_list.append(self_k_cache_updated)
            self_v_cache_update_list.append(self_v_cache_updated)
            i += 1

        n_layer_self_k_cache_updated = torch.stack(self_k_cache_update_list)
        n_layer_self_v_cache_updated = torch.stack(self_v_cache_update_list)

        x = self.textDecoder.ln(x) #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same

        return logits, n_layer_self_k_cache_updated, n_layer_self_v_cache_updated
        #TODO get probs for ONNX export!!!

class TextDecoderPreKv1tkn(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder):
        super().__init__()
        self.textDecoder = in_textDecoder

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block))

    def forward(self, x: Tensor, k: Tensor, v: Tensor, offset: Tensor):
        pos_emb_slice = self.textDecoder.positional_embedding[offset] #diff! For onnx export
        x = self.textDecoder.token_embedding(x) + pos_emb_slice #same
        x = x.to(k.dtype) #same

        i = 0
        for block in self.blocks:
            x = block(x, k[i], v[i], mask=self.textDecoder.mask) # diff!
            i += 1

        x = self.textDecoder.ln(x) #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same

        probs = F.softmax(logits, dim=-1)

        return probs

class TextDecoderPreKv16tkn(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder):
        super().__init__()
        self.textDecoder = in_textDecoder

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block))

    def forward(self, x: Tensor, k: Tensor, v: Tensor):
        pos_emb_slice = self.textDecoder.positional_embedding[0:16] #diff!
        x = self.textDecoder.token_embedding(x) + pos_emb_slice #same
        x = x.to(k.dtype) #same

        i = 0
        for block in self.blocks:
            x = block(x, k[i], v[i], mask=self.textDecoder.mask) # diff!
            i += 1

        x = self.textDecoder.ln(x) #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same

        #TODO もっと早くSliceすべきかも
        probs = F.softmax(logits[:,15:16,:], dim=-1)

        return probs

class TextDecoderPreKvNtkn(nn.Module):
    def __init__(self, in_textDecoder: TextDecoder, in_n_token: int):
        super().__init__()
        self.textDecoder = in_textDecoder
        self.n_token = in_n_token

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlock_KvCache(orginal_block))

    def forward(self, x: Tensor, k: Tensor, v: Tensor):
        pos_emb_slice = self.textDecoder.positional_embedding[0:self.n_token] #diff!
        x = self.textDecoder.token_embedding(x) + pos_emb_slice #same
        x = x.to(k.dtype) #same

        i = 0
        for block in self.blocks:
            x = block(x, k[i], v[i], mask=self.textDecoder.mask) # diff!
            i += 1

        # get only last token's logit
        x = x[:,self.n_token-1:self.n_token,:] #diff!

        x = self.textDecoder.ln(x) #same
        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float() #same

        probs = F.softmax(logits, dim=-1) #diff!

        return probs

class WhisperPreKV(nn.Module):
    def __init__(self, in_whisper: Whisper):
        super().__init__()
        self.whisper = in_whisper
        self.dims = self.whisper.dims
        self.encoder = AudioEncoder_KvCache(self.whisper.encoder, self.whisper.decoder, 1500, 0) #TODO まずはキャッシュやめて動かす
        self.decoder = TextDecoder_KvCache(self.whisper.decoder, 0, 0)  #TODO まずは動的にn_ctx変えて動かす
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

    def exportOnnxEncoder(self, name):
        device = self.whisper.device
        dummy_input = torch.randn((1, 3000, 80), dtype=torch.float32).to(device)
        input_names = ["mel_t"]
        output_names = ['keys', 'values']
        file_name = "encoder_" + name + ".onnx"
        torch.onnx.export(self.encoder,
                        ( dummy_input ),
                        file_name,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names
        )
        onnx_model = onnx.load(f'{file_name}')
        onnx_model_simp, check = simplify(onnx_model)
        file_name_simp =  "encoder_" + name + ".smpl.onnx"
        onnx.save(onnx_model_simp, f'{file_name_simp}')

    def exportOnnxDecoder(self, name): # n_token = 1
        n_state = self.dims.n_text_state
        n_layer = self.dims.n_text_layer

        token_list = []
        token_list.append(torch.tensor(0, dtype=torch.int64).to('cuda'))
        dummy_tokens = torch.stack(token_list).unsqueeze(0)
        dummy_k = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to('cuda')
        dummy_v = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to('cuda')
        dummy_offset = torch.tensor(0, dtype=torch.int64).to('cuda').unsqueeze(0)

        inputs = ( dummy_tokens, dummy_k, dummy_v, dummy_offset )

        input_names = ['tokens', 'keys', 'values', 'offset']
        #output_names = ['logits']
        output_names = ['probabilities']
        file_name = "decoder_" + name + "_1tkn.onnx"
        torch.onnx.export(self.decoder1tkn,
                        inputs,
                        file_name,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names,
                        #dynamic_axes={'kv_cache_in': {1: 'kv_cacheIn_dynamic_axes_1',
                        #                              2: 'kv_cacheIn_dynamic_axes_2'},
                        #              'kv_cache': {1: 'kv_cache_dynamic_axes_1',
                        #                           2: 'kv_cache_dynamic_axes_2'}
                        #              }
                        )
        onnx_model = onnx.load(f'{file_name}')
        onnx_model_simp, check = simplify(onnx_model)
        file_name_simp =  "decoder_" + name + "_1tkn.smpl.onnx"
        onnx.save(onnx_model_simp, f'{file_name_simp}')

    def exportOnnxDecoder16tkn(self, name, n_token: int):
        n_state = self.dims.n_text_state
        n_layer = self.dims.n_text_layer

        token_list = []
        for i in range(n_token):
            token_list.append(torch.tensor(0, dtype=torch.int64).to('cuda'))
        dummy_tokens = torch.stack(token_list).unsqueeze(0)
        dummy_k = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to('cuda')
        dummy_v = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to('cuda')
        dummy_offset = torch.tensor(0, dtype=torch.int64).to('cuda')

        inputs = ( dummy_tokens, dummy_k, dummy_v )

        input_names = ['tokens', 'keys', 'values']
        #output_names = ['logits']
        #output_names = ['no_speech_prob, probabilities']
        output_names = ['probabilities']
        file_name = "decoder_" + name + "_" + str(n_token) + "tkn.onnx"
        torch.onnx.export(self.decoder16tkn,
                        inputs,
                        file_name,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names,
                        #dynamic_axes={'kv_cache_in': {1: 'kv_cacheIn_dynamic_axes_1',
                        #                              2: 'kv_cacheIn_dynamic_axes_2'},
                        #              'kv_cache': {1: 'kv_cache_dynamic_axes_1',
                        #                           2: 'kv_cache_dynamic_axes_2'}
                        #              }
                        )
        onnx_model = onnx.load(f'{file_name}')
        onnx_model_simp, check = simplify(onnx_model)
        file_name_simp =  "decoder_" + name + "_" + str(n_token) + "tkn.smpl.onnx"
        onnx.save(onnx_model_simp, f'{file_name_simp}')

    def exportOnnxDecoderNtkn(self, name, n_token: int):
        self.decoderNtkn = TextDecoderPreKvNtkn(self.whisper.decoder, n_token)
        device = self.whisper.device

        n_state = self.dims.n_text_state
        n_layer = self.dims.n_text_layer

        token_list = []
        for i in range(n_token):
            token_list.append(torch.tensor(0, dtype=torch.int64).to(device))
        dummy_tokens = torch.stack(token_list).unsqueeze(0)
        dummy_k = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to(device)
        dummy_v = torch.randn((n_layer, 1, 1500, n_state), dtype=torch.float32).to(device)
        dummy_offset = torch.tensor(0, dtype=torch.int64).to(device)

        inputs = ( dummy_tokens, dummy_k, dummy_v )

        input_names = ['tokens', 'keys', 'values']
        #output_names = ['logits']
        #output_names = ['no_speech_prob, probabilities']
        output_names = ['probabilities']
        file_name = "decoder_" + name + "_" + str(n_token) + "tkn.onnx"
        torch.onnx.export(self.decoderNtkn,
                        inputs,
                        file_name,
                        export_params=True,
                        opset_version=12,
                        do_constant_folding=True,
                        input_names=input_names, 
                        output_names=output_names,
                        #dynamic_axes={'kv_cache_in': {1: 'kv_cacheIn_dynamic_axes_1',
                        #                              2: 'kv_cacheIn_dynamic_axes_2'},
                        #              'kv_cache': {1: 'kv_cache_dynamic_axes_1',
                        #                           2: 'kv_cache_dynamic_axes_2'}
                        #              }
                        )
        onnx_model = onnx.load(f'{file_name}')
        onnx_model_simp, check = simplify(onnx_model)
        file_name_simp =  "decoder_" + name + "_" + str(n_token) + "tkn.smpl.onnx"
        onnx.save(onnx_model_simp, f'{file_name_simp}')