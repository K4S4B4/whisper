import onnx
from scs4onnx import shrinking

def shrink(file_name):
    src_onnx = file_name + '.onnx'
    opt_onnx = file_name + '_scs.onnx'

    shrinking(
      input_onnx_file_path=src_onnx,
      output_onnx_file_path=opt_onnx,
      mode='shrink',
      non_verbose=False
    )

#optimize('encoder_mask_1500_0_tiny')
shrink('decoder_sl_a16_1_-1_16_1500_int32_tiny_opt_fp16')