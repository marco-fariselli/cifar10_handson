from nntool.api import NNGraph
import tensorflow as tf
from keras import datasets
from keras.utils import to_categorical
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)
from nntool.api.utils import quantization_options, model_settings
from tqdm import tqdm
import pandas as pd

USE_NE16 = False
HWC = False
L1_SIZE = 64000
L2_SIZE = int(200*1024)

def representative_dataset(hwc=HWC or USE_NE16):
	for input_tensor in tqdm(CALIBRATION_IMGS):
		if not hwc:
			input_tensor = input_tensor.transpose(2, 0, 1)
		yield input_tensor

def nntool_accuracy(graph, test_images, test_labels, hwc=HWC or USE_NE16, quantize=True):
    predictions = np.zeros((len(test_images),), dtype=int)
    for i, (test_image, test_label) in enumerate(tqdm(zip(test_images, test_labels), total=len(test_labels))):
        if not hwc:
            test_image = test_image.transpose((2,0,1))
        output = graph.execute([test_image], dequantize=quantize)
        predictions[i] = output[-1][0].argmax()

    test_labels_not_one_hot = np.argmax(test_labels, 1)
    accuracy = (np.sum(test_labels_not_one_hot == predictions) * 100) / len(test_images)
    return accuracy


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, train_labels = train_images[:10000], train_labels[:10000]
test_images, test_labels = test_images[:1000], test_labels[:1000]

# Converting the pixels data to float type
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Standardizing (255 is the total number of pixels an image can have)
train_images = (train_images / 128) - 1.0
test_images = (test_images / 128) - 1.0

# One hot encoding the target class (labels)
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


model_perf = pd.DataFrame()
for model_name in ["v1", "v2", "v3", "v4"]:
	checkpoint_path = f"./checkpoints/saved_model_{model_name}/"
	G = NNGraph.load_graph(f"{checkpoint_path}/cifar10_model_{model_name}_fp32.tflite", load_quantization=False)
	#G.draw(filepath="draw", view=True)
	max_activ_size, total_params = G.total_memory_usage
	ops = G.total_ops

	print(f"{G.name}:")
	print(f"\tMax Active Size:\t{max_activ_size} elements")
	print(f"\tTotal # Parameters:\t{total_params} elements")
	print(f"\tTotal # Operations:\t{ops / 1e6:.2f} MOps")
	G.adjust_order()
	G.fusions('scaled_match_group')
	G.fusions('expression_matcher')
	CALIBRATION_IMGS = train_images[:50]

	print("Calibrating...")
	stats = G.collect_statistics(representative_dataset())
	G.quantize(
	    statistics=stats,
	    graph_options=quantization_options(
			use_ne16=USE_NE16,
	        hwc=USE_NE16
		),
	)
	print("Testing....")
	nntool_float_accuracy = nntool_accuracy(G, test_images[:100], test_labels[:100], quantize=False)
	print(f"Full Prec model accuracy: {nntool_float_accuracy}")
	nntool_quant_accuracy = nntool_accuracy(G, test_images[:100], test_labels[:100])
	print(f"Quantized model accuracy: {nntool_quant_accuracy}")

	# Autotiler options: make the autotiler allocate the input of the network and reuse that space after the first layer
	# more L2 for the rest of the network
	G[0].at_options.allocate = 1
	G[0].at_options
	res = G.execute_on_target(
	    pmsis_os='freertos',
	    platform="gvsoc",
	    directory="test_run",
	    output_tensors=0,
	    at_log=True,
	    settings=model_settings(
	        l1_size=L1_SIZE,
	        l2_size=L2_SIZE, 
	        tensor_directory='./tensors',
	        graph_const_exec_from_flash=True,
		),
	    at_loglevel=1,
		progress=True,
	)
	model_perf[model_name] = {
		"full_prec_acc": nntool_float_accuracy,
		"quant_acc": nntool_quant_accuracy,
		"cyc": res.performance[-1][1],
		"op": res.performance[-1][2],
		"op/cyc": res.performance[-1][3],
		"tot_params": total_params
	}
	print(model_perf.T)

############# FINAL RESULTS ##############
#'v1': {'full_prec_acc': 68.0, 'quant_acc': 68.0, 'cyc': 8931716, 'op': 37550816, 'op/cyc': 4.20, 'tot_params': 1438522}
#'v2': {'full_prec_acc': 68.0, 'quant_acc': 69.0, 'cyc': 6649951, 'op': 30844298, 'op/cyc': 4.63, 'tot_params':  569370}
#'v3': {'full_prec_acc': 57.0, 'quant_acc': 58.0, 'cyc': 2203115, 'op': 10104970, 'op/cyc': 4.58, 'tot_params':  253978}
#'v5': {'full_prec_acc': 65.0, 'quant_acc': 65.0, 'cyc':  815722, 'op':  2807178, 'op/cyc': 3.44, 'tot_params':  297754}
