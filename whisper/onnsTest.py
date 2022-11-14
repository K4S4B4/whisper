import onnxruntime
import numpy
import time

sess_options = onnxruntime.SessionOptions()

#sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

providers = ['DmlExecutionProvider']
providers = ['CUDAExecutionProvider']

sess_options.log_severity_level = 0
sess_options.log_verbosity_level = 1

#model_path = 'decoder_staticLoop_32_64_base.onnx'
#model_path = 'decoder_staticLoop_8_4_base.onnx'
model_path = 'decoder_staticLoop_8_8_base_smpl.onnx'
#model_path = 'decoder_staticLoop_8_2_small_smpl.onnx'

load_start = time.time()
session = onnxruntime.InferenceSession(model_path, sess_options, providers)
print("Load took:", time.time() - load_start)

n_state = 512
n_ctx_in = 8

total_samples=10
latency = []
lengths = []

for i in range(total_samples):
    in_tokens = numpy.ones((1,n_ctx_in), dtype='int64')
    audio_feature = numpy.ones((1, 1500, n_state), dtype='float32')

    opt_inputs = {
        'in_tokens':  in_tokens,
        'audio_feature': audio_feature
    }

    start = time.time()
    opt_outputs = session.run(None, opt_inputs)
    latency.append(time.time() - start)

print("OnnxRuntime Inference time with actual sequence length = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))