import onnx
import onnxoptimizer

#OPTIONS = [
#    'eliminate_nop_cast',
#    'eliminate_nop_dropout',
#    'eliminate_nop_flatten',
#    'extract_constant_to_initializer',
#    'eliminate_if_with_const_cond',
#    'eliminate_nop_monotone_argmax',
#    'eliminate_nop_pad',
#    'eliminate_nop_concat',
#    'eliminate_nop_split',
#    'eliminate_nop_expand',
#    'eliminate_shape_gather',
#    'eliminate_slice_after_shape',
#    'eliminate_nop_transpose',
#    'fuse_add_bias_into_conv',
#    'fuse_bn_into_conv',
#    'fuse_consecutive_concats',
#    'fuse_consecutive_log_softmax',
#    'fuse_consecutive_reduce_unsqueeze',
#    'fuse_consecutive_squeezes',
#    'fuse_consecutive_transposes',
#    'fuse_matmul_add_bias_into_gemm',
#    'fuse_pad_into_conv',
#    'fuse_pad_into_pool',
#    'fuse_transpose_into_gemm',
#    'replace_einsum_with_matmul',
#    'lift_lexical_references',
#    'split_init',
#    'split_predict',
#    'fuse_concat_into_reshape',
#    'eliminate_nop_reshape',
#    'eliminate_deadend',
#    'eliminate_identity',
#    'eliminate_shape_op',
#    'eliminate_unused_initializer',
#    'eliminate_duplicate_initializer'
#]

def printOptim():
    all_passes = onnxoptimizer.get_available_passes()
    print("Available optimization passes:")
    for p in all_passes:
        print(p)
    print()

def optimize(file_name):
    src_onnx = file_name + '.onnx'
    opt_onnx = file_name + '_ofopt.onnx'

    # load model
    model = onnx.load(src_onnx)

    # optimize
    model = onnxoptimizer.optimize(model, ['fuse_matmul_add_bias_into_gemm'])

    # save optimized model
    #onnx.save(model, opt_onnx)

    with open(opt_onnx, "wb") as f:
        f.write(model.SerializeToString())

#optimize('encoder_mask_1500_0_tiny')
shrink('decoder_sl_a16_1_16_16_1500_int32_small_opt_fp16')