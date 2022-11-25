import torch
#import onnx

from modelStaticLoop import TextDecoder_dynamicLoop

class TextDecoder_dynamicLoop_tiny_body(TextDecoder_dynamicLoop):
    def forward(self, trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count,
                in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3,
                in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3,
                in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3,
    ):
        self_kv_list =    [in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3]
        n_layer_cross_k = [in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3]
        n_layer_cross_v = [in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3]

        out_cond, out_token, out_positions, out_token_history, out_repeat_count, out_self_kv_list = self.loopBody(trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count, self_kv_list, n_layer_cross_k, n_layer_cross_v)

        out_self_kv0,out_self_kv1,out_self_kv2,out_self_kv3 = tuple(out_self_kv_list)

        return (
            out_cond, 
            out_token, out_positions, 
            out_token_history, out_repeat_count,
            out_self_kv0,out_self_kv1,out_self_kv2,out_self_kv3,
            in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3,
            in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3,
        )

class TextDecoder_dynamicLoop_tiny_driver(TextDecoder_dynamicLoop):
    def forward(self, 
                in_tokens, #[n, n_ctx_in]
                in_positions, #[n, n_ctx_in]
                xa
                ):

        out_token_history = torch.zeros([1,1], dtype=torch.int64).to(self.device)

        (
            trip_count, in_cond, #in_tokens, in_positions, 
            in_token_history, in_repeat_count,
            in_self_kv_list,
            n_layer_cross_k,
            n_layer_cross_v,
        ) = self.preprocess(in_tokens, xa)

        in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3 = tuple(in_self_kv_list)
        in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3 = tuple(n_layer_cross_k)
        in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3 = tuple(n_layer_cross_v)

        return (
                out_token_history,
                trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count,
                in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, 
                in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, 
                in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, 
        )
    
class TextDecoder_dynamicLoop_base_body(TextDecoder_dynamicLoop):
    def forward(self, trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count,
                in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, in_self_kv4, in_self_kv5,
                in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5,
                in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5,
    ):
        self_kv_list =    [in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, in_self_kv4, in_self_kv5]
        n_layer_cross_k = [in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5]
        n_layer_cross_v = [in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5]

        out_cond, out_token, out_positions, out_token_history, out_repeat_count, out_self_kv_list = self.loopBody(trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count, self_kv_list, n_layer_cross_k, n_layer_cross_v)

        out_self_kv0,out_self_kv1,out_self_kv2,out_self_kv3,out_self_kv4,out_self_kv5 = tuple(out_self_kv_list)

        return (
            out_cond, 
            out_token, out_positions, 
            out_token_history, out_repeat_count,
            out_self_kv0,out_self_kv1,out_self_kv2,out_self_kv3,out_self_kv4,out_self_kv5,
            in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5,  
            in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5, 
        )

class TextDecoder_dynamicLoop_base_driver(TextDecoder_dynamicLoop):
    def forward(self, 
                in_tokens, #[n, n_ctx_in]
                in_positions, #[n, n_ctx_in]
                xa
                ):
        (
            out_token_scan,
            trip_count, in_cond,
            in_self_kv_list,
            n_layer_cross_k,
            n_layer_cross_v,
        ) = self.preprocess(in_tokens, xa)

        in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, in_self_kv4, in_self_kv5 = tuple(in_self_kv_list)
        in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5 = tuple(n_layer_cross_k)
        in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5 = tuple(n_layer_cross_v)

        return (out_token_scan,
                trip_count, in_cond, in_tokens, in_positions, 
                in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, in_self_kv4, in_self_kv5,
                in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5,
                in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5,
        )

class TextDecoder_dynamicLoop_small_body(TextDecoder_dynamicLoop):
    def forward(self, trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count,
                in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, in_self_kv4, in_self_kv5, in_self_kv6, in_self_kv7, in_self_kv8, in_self_kv9, in_self_kv10, in_self_kv11,
                in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5, in_cross_k6, in_cross_k7, in_cross_k8, in_cross_k9, in_cross_k10, in_cross_k11,
                in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5, in_cross_v6, in_cross_v7, in_cross_v8, in_cross_v9, in_cross_v10, in_cross_v11,
    ):
        self_kv_list =    [in_self_kv0, in_self_kv1, in_self_kv2, in_self_kv3, in_self_kv4, in_self_kv5, in_self_kv6, in_self_kv7, in_self_kv8, in_self_kv9, in_self_kv10, in_self_kv11,]
        n_layer_cross_k = [in_cross_k0, in_cross_k1, in_cross_k2, in_cross_k3, in_cross_k4, in_cross_k5, in_cross_k6, in_cross_k7, in_cross_k8, in_cross_k9, in_cross_k10, in_cross_k11,]
        n_layer_cross_v = [in_cross_v0, in_cross_v1, in_cross_v2, in_cross_v3, in_cross_v4, in_cross_v5, in_cross_v6, in_cross_v7, in_cross_v8, in_cross_v9, in_cross_v10, in_cross_v11,]

        out_cond, out_token, out_positions, out_token_history, out_repeat_count, out_self_kv_list = self.loopBody(trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count, self_kv_list, n_layer_cross_k, n_layer_cross_v)

        outputList = [out_cond, out_token, out_positions, out_token_history, out_repeat_count]
        outputList += out_self_kv_list
        outputList += n_layer_cross_k
        outputList += n_layer_cross_v

        return tuple(outputList)

class TextDecoder_dynamicLoop_small_driver(TextDecoder_dynamicLoop):
    def forward(self, 
                in_tokens, #[n, n_ctx_in]
                in_positions, #[n, n_ctx_in]
                xa
                ):

        out_token_history = torch.zeros([1,1], dtype=torch.int64).to(self.device)

        (
            trip_count, in_cond, #in_tokens, in_positions, 
            in_token_history, in_repeat_count,
            in_self_kv_list,
            n_layer_cross_k,
            n_layer_cross_v,
        ) = self.preprocess(in_tokens, xa)

        outputList = [out_token_history, trip_count, in_cond, in_tokens, in_positions, in_token_history, in_repeat_count]
        outputList += in_self_kv_list
        outputList += n_layer_cross_k
        outputList += n_layer_cross_v

        return tuple(outputList)
    
def export_TextDecoder_dynamicLoop_driver(name, model, n_ctx_in: int, n_ctx_out: int, n_audioFeature: int):
    isMultilingual = not name.endswith('en')
    device = model.whisper.device
    n_state = model.whisper.dims.n_text_state
    n_layer = model.whisper.dims.n_text_layer
    n_batch = 1

    if name.startswith('tiny'):
        decoder = TextDecoder_dynamicLoop_tiny_driver(model.whisper.decoder, n_ctx_out, isMultilingual, n_layer)
    if name.startswith('base'):
        decoder = TextDecoder_dynamicLoop_base_driver(model.whisper.decoder, n_ctx_out, isMultilingual, n_layer)
    if name.startswith('small'):
        decoder = TextDecoder_dynamicLoop_small_driver(model.whisper.decoder, n_ctx_out, isMultilingual, n_layer)

    input_names = ['in_tokens', 'in_positions', 'audio_feature']
    output_names = ['out_token_history', 'trip_count', 'in_cond', 'in_tokens_dum', 'in_positions_dum', 'in_token_history', 'in_repeat_count']
    for i in range(n_layer):
        output_names.append('in_self_kv' + str(i))
    for i in range(n_layer):
        output_names.append('in_cross_k' + str(i))
    for i in range(n_layer):
        output_names.append('in_cross_v' + str(i))

    file_base = "decoder_dl_a16_"
    file_base += str(n_ctx_in) + "_"
    file_base += str(n_ctx_out) + "_"
    file_base += str(n_audioFeature) + "_"
    file_base += name + "_"
    file_base += "driver"
    file_onnx = file_base + ".onnx"

    dynamic_axes = dict()
    if n_ctx_in < 0:
        dynamic_axes['in_tokens'] = {1: 'n_tokens_in'}
        dynamic_axes['in_positions'] = {1: 'n_tokens_in'}
        n_ctx_in = 4
    if n_audioFeature < 0:
        dynamic_axes['audio_feature'] = {1: 'n_audio_feature'}
        n_audioFeature = 1500
    dynamic_axes['in_token_history'] = {1: 'n_tokens_history_in'}
    dynamic_axes['out_token_history'] = {1: 'n_tokens_history_out'}
    for i in range(n_layer):
        dynamic_axes['in_self_kv' + str(i)] = {3: 'n_ctx_cache_in'}

    token_list = []
    for i in range(n_ctx_in):
        token_list.append(torch.tensor(i, dtype=torch.int64).to(device))
    dummy_tokens = torch.stack(token_list).unsqueeze(0)
    dummy_tokens = dummy_tokens.expand(n_batch, n_ctx_in)

    dummy_positions = torch.arange(0, n_ctx_in, 1, dtype=torch.int32).unsqueeze(0).to(device)
    dummy_audioFeature = torch.randn((1, n_audioFeature, n_state), dtype=torch.float32).to(device)

    inputs = ( dummy_tokens, dummy_positions, dummy_audioFeature )

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

def export_TextDecoder_dynamicLoop_body(name, model, n_ctx_in: int, n_ctx_out: int, n_audioFeature: int):
    isMultilingual = not name.endswith('en')
    device = model.whisper.device
    n_state = model.whisper.dims.n_text_state
    n_layer = model.whisper.dims.n_text_layer
    n_head = model.whisper.dims.n_text_head
    n_batch = 1

    if name.startswith('tiny'):
        decoder = TextDecoder_dynamicLoop_tiny_body(model.whisper.decoder, n_ctx_out, isMultilingual, n_layer)
    if name.startswith('base'):
        decoder = TextDecoder_dynamicLoop_base_body(model.whisper.decoder, n_ctx_out, isMultilingual, n_layer)
    if name.startswith('small'):
        decoder = TextDecoder_dynamicLoop_small_body(model.whisper.decoder, n_ctx_out, isMultilingual, n_layer)

    input_names = ['trip_count', 'in_cond', 'in_tokens', 'in_positions', 'in_token_history', 'in_repeat_count']
    for i in range(n_layer):
        input_names.append('in_self_kv' + str(i))
    for i in range(n_layer):
        input_names.append('in_cross_k' + str(i))
    for i in range(n_layer):
        input_names.append('in_cross_v' + str(i))
    output_names = ['out_cond', 'out_tokens', 'out_positions', 'out_token_history', 'out_repeat_count']
    for i in range(n_layer):
        output_names.append('out_self_kv' + str(i))
    for i in range(n_layer):
        output_names.append('out_cross_k' + str(i))
    for i in range(n_layer):
        output_names.append('out_cross_v' + str(i))
    #output_names.append('out_token_scan')

    file_base = "decoder_dl_a16_"
    file_base += str(n_ctx_in) + "_"
    file_base += str(n_ctx_out) + "_"
    file_base += str(n_audioFeature) + "_"
    file_base += name + "_"
    file_base += "body"
    file_onnx = file_base + ".onnx"

    dynamic_axes = dict()
    if n_ctx_in < 0:
        dynamic_axes['in_tokens'] = {1: 'n_tokens_in'}
        dynamic_axes['in_positions'] = {1: 'n_tokens_in'}
        n_ctx_in = 4
    if n_audioFeature < 0:
        dynamic_axes['audio_feature'] = {1: 'n_audio_feature'}
        n_audioFeature = 1500
    for i in range(n_layer):
        dynamic_axes['in_self_kv' + str(i)] = {3: 'n_ctx_cache_in'}
        dynamic_axes['out_self_kv' + str(i)] = {3: 'n_ctx_cache_out'}
    dynamic_axes['in_token_history'] = {1: 'n_tokens_history_in'}
    dynamic_axes['out_token_history'] = {1: 'n_tokens_history_out'}

    trip_count = torch.tensor(n_ctx_out, dtype=torch.int64).to(device) #max 128 loop
    in_cond = torch.tensor(True, dtype=torch.bool).to(device)

    token_list = []
    for i in range(n_ctx_in):
        token_list.append(torch.tensor(i, dtype=torch.int64).to(device))
    dummy_tokens = torch.stack(token_list).unsqueeze(0)
    dummy_tokens = dummy_tokens.expand(n_batch, n_ctx_in)

    dummy_positions = torch.arange(0, n_ctx_in, 1, dtype=torch.int32).unsqueeze(0).to(device)
    dummy_token_history = dummy_tokens
    dummy_repeat_count = torch.zeros([1,1], dtype=torch.int64).to(device)

    inputsList = [trip_count, in_cond, dummy_tokens, dummy_positions, dummy_token_history, dummy_repeat_count]

    for i in range(n_layer):
        inputsList.append(torch.zeros([2, 1, n_head, 3, 64], dtype=torch.float32).to(device))
    for i in range(n_layer):
        inputsList.append(torch.zeros([1, n_head, 64, n_audioFeature], dtype=torch.float32).to(device))
    for i in range(n_layer):
        inputsList.append(torch.zeros([1, n_head, n_audioFeature, 64], dtype=torch.float32).to(device))

    inputs = tuple(inputsList)
    torch.onnx.export(decoder,
                    inputs,
                    file_onnx,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=False,
                    input_names=input_names, 
                    output_names=output_names,
                    dynamic_axes=dynamic_axes
    )

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
    #export_TextDecoder_dynamicLoop_driver(model_name, model, -1, 128, 1500)
    export_TextDecoder_dynamicLoop_body(model_name, model, -1, 128, 1500)

    #export_TextDecoder_StaticLoop(model_name, model, 8, 2, True)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 4)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 8)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 16)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 32)
    #export_TextDecoder_StaticLoop(model_name, model, 8, 64)
    #export_TextDecoder_StaticLoop(model_name, model, 32, 64)

    ## forced alignment
    #export_TextDecoder_StaticLoop(model_name, model, 8, -1, 0, 1500, True, 32)
    #export_TextDecoder_StaticLoop(model_name, model, 4, 32, 0, 1500, True, 32)

if __name__ == '__main__':
    #model_name = "tiny"
    #model_name = "base"
    #model_name = "small"
    #model_name = "medium"
    #model_name = "tiny.en"
    #model_name = "base.en"
    #model_name = "small.en"
    #model_name = "medium.en"

    #executeExport("tiny")
    #executeExport("base")
    executeExport("small")
    #executeExport("medium")
    #executeExport("tiny.en")
    #executeExport("base.en")
    #executeExport("small.en")
    #executeExport("medium.en")

    #executeSimplify(model_name)