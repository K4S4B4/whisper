import onnxruntime
#import openvino.utils as utils
import time
import numpy as np

#utils.add_openvino_libs_to_path()

def execute(modelPath, providers, w, h, optimLevel):

	sess_options = onnxruntime.SessionOptions()
	#sess_options.enable_profiling = True
	sess_options.log_severity_level = 0
	sess_options.log_verbosity_level = 1
	sess_options.graph_optimization_level = optimLevel
	#sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
	#sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

	session = onnxruntime.InferenceSession(modelPath, sess_options, providers, [{"device_type" : "CPU_FP32"}],)

	input = session.get_inputs()[0]
	input_name = input.name
	cap = cv2.VideoCapture(1)
	_, frame = cap.read()
	inputImg = cv2.resize(frame, dsize=(w,h))
	if (input.type == 'tensor(uint8)'):
		inputImg = np.expand_dims(inputImg, axis=0).astype(np.uint8)
	elif (input.type == 'tensor(int32)'):
		inputImg = np.expand_dims(inputImg, axis=0).astype(np.int32)
	elif (input.type == 'tensor(float)'):
		inputImg = np.expand_dims(inputImg, axis=0).astype(np.float32)
	else:
		inputImg = np.expand_dims(inputImg, axis=0).astype(np.int8)

	#inputImg = np.transpose(inputImg, axes = (0,3,2,1))

	ortInput = {input_name: inputImg}
	ortOut = session.run(None, ortInput)

	start = time.time()
	count = 0
	while True:
		ortInput = {input_name: inputImg}
		ortOut = session.run(None, ortInput)

		count = count +1

		if count == 20:
			break

	dt = (time.time() - start) * 50
	return dt

def run(modelPath, providers, w, h):
	d = execute(modelPath, providers, w, h, onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
	b = execute(modelPath, providers, w, h, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC)
	e = execute(modelPath, providers, w, h, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
	a = execute(modelPath, providers, w, h, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
	return (d, b, e, a)


def runModels(providers):
    print(providers)

    #modelPath = 'encoder_256_-1_base.smpl.onnx'
    #modelPath = 'encoder_-1_1000_base_smpl.onnx'
    modelPath = 'encoder_-1_-1_base_smpl.onnx'
    w,h=224,224
    dt = run(modelPath, providers, w, h)
    print("Hand Full", dt)

    modelPath = 'Models/hand_landmark_lite.BGR_Byte.Normalized.onnx'
    w,h=224,224
    dt = run(modelPath, providers, w, h)
    print("Hand Lite", dt)

    #modelPath = 'Models/movenet_lightning_v4.onnx'
    #modelPath = 'Models/movenet_lightning_v4_nchw.onnx'
    modelPath = 'Models/movenet_lightning_v4_1x192x192x3xBGRxByte_upper.onnx'
    #modelPath = 'Models/movenet_lightning_v4_1x192x192x3xBGRxByte_upper.smpl.onnx'
    #modelPath = 'Models/movenet_lightning_v4_1x192x192x3xBGRxByte_upper.smpl.optm.onnx'
    w,h=192,192
    dt = run(modelPath, providers, w, h)
    print("Mvnt Lite", dt)

    modelPath = 'Models/model_float32_BHWC_BgrByte.optmzd.onnx' #face
    w,h=192,192
    dt = run(modelPath, providers, w, h)
    print("Face attn", dt)

    modelPath = 'Models/pose_landmark_full.onnx' #blaze pose
    w,h=256,256
    dt = run(modelPath, providers, w, h)
    print("BlPs Full", dt)

    modelPath = 'Models/pose_landmark_lite.onnx' #blaze pose
    w,h=256,256
    dt = run(modelPath, providers, w, h)
    print("BlPs Lite", dt)

#runModels(['CPUExecutionProvider'])
#runModels(['OpenVINOExecutionProvider'])
runModels(['DmlExecutionProvider'])