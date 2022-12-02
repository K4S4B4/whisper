import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize(file_name):
    src_onnx = file_name + '.onnx'
    opt_onnx = file_name + '_qtz.onnx'
    quantize_dynamic(src_onnx, opt_onnx, nodes_to_exclude=['/audioEncoder/conv1/Conv', '/audioEncoder/conv2/Conv'])

quantize('encoder_mask_1500_0_tiny_opt')
quantize('encoder_mask_1500_0_small_opt')