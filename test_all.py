from nntool.api import NNGraph
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
logging.basicConfig(level=logging.ERROR)
from nntool.reports.graph_reporter import graph_walk
from nntool.graph.types import ConstantInputNode
import math
from tqdm import tqdm

def nntool_inference(graph, test_images, test_labels, hwc=False):
    predictions = np.zeros((len(test_images),), dtype=int)
    for i, (test_image, test_label) in enumerate(tqdm(zip(test_images, test_labels), total=len(test_labels))):
        if not hwc:
            test_image = test_image.transpose((2,0,1))
        output = graph.execute([test_image], dequantize=True)
        predictions[i] = output[-1][0].argmax()

    test_labels_not_one_hot = np.argmax(test_labels, 1)
    accuracy = (np.sum(test_labels_not_one_hot == predictions) * 100) / len(test_images)
    return accuracy

def get_graph_memory_usage(steps, liveness, quantization_set=None):
    max_active = 0
    tot_params = 0
    for _, node, active, params_size, _ in graph_walk(steps, liveness):
        if isinstance(node, ConstantInputNode) and node.use_compressed:
            bits_per_element = node.compressed_value.bits
        elif quantization_set:
            bits_per_element = quantization_set[node.name].out_qs[0].bits
        else:
            bits_per_element = 8
        tot_params += (params_size * bits_per_element)
        if active > max_active:
            max_active = active
    return max_active, math.ceil(tot_params/8)

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


model_perf = {}
for model_name in ["v1", "v2", "v3", "v5"]:
	G = NNGraph.load_graph(f"cifar10_model_{model_name}_fp32.tflite", load_quantization=False)
	#G.draw(filepath="draw", view=True)
	max_activ_size, total_params = get_graph_memory_usage(G.graph_state.steps, G.graph_state.liveness)
	ops = G.total_ops

	print(f"{G.name}:")
	print(f"\tMax Active Size:\t{max_activ_size} elements")
	print(f"\tTotal # Parameters:\t{total_params} elements")
	print(f"\tTotal # Operations:\t{ops / 1e6:.2f} MOps")
	G.adjust_order()
	G.fusions('scaled_match_group')
	G.fusions('expression_matcher')
	CALIBRATION_IMGS = train_images[:50]
	def representative_dataset(hwc=False):
	    for input_tensor in tqdm(CALIBRATION_IMGS):
	        if not hwc:
	            input_tensor = input_tensor.transpose(2, 0, 1)
	        yield input_tensor

	print("Calibrating...")
	stats = G.collect_statistics(representative_dataset())
	G.quantize(
	    statistics=stats,
	    graph_options={
	        'bits': 8,
	        'use_ne16': False,
	        'hwc': False
	    },
	)
	print("Testing....")
	nntool_quant_accuracy = nntool_inference(G, test_images[:100], test_labels[:100])
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
	    dont_run=False,
	    do_clean=False,
	    settings={
	        'l1_size': 64000,
	        'l2_size': 200000, 
	        'tensor_directory': './tensors',
	        'graph_const_exec_from_flash': True,
	    },
	    #cmake=False,
	    at_loglevel=1,
	)
	for l in res.at_log:
	    print(l)
	model_perf[model_name] = {"acc": nntool_quant_accuracy, "cyc": res.performance[-1][1], "op": res.performance[-1][2], "op/cyc": res.performance[-1][3], "tot_params": total_params}
	print(model_perf)

############# FINAL RESULTS ##############
#'v1': {'acc': 68.0, 'cyc': 8931716, 'op': 37550816, 'op/cyc': 4.20, 'tot_params': 1438522}
#'v2': {'acc': 69.0, 'cyc': 6649951, 'op': 30844298, 'op/cyc': 4.63, 'tot_params':  569370}
#'v3': {'acc': 58.0, 'cyc': 2203115, 'op': 10104970, 'op/cyc': 4.58, 'tot_params':  253978}
#'v5': {'acc': 65.0, 'cyc':  815722, 'op':  2807178, 'op/cyc': 3.44, 'tot_params':  297754}
