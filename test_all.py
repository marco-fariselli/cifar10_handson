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

for model_name in ["v1", "v2", "v3"]:
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
	nntool_quant_accuracy = nntool_inference(G, test_images[:1000], test_labels[:1000])
	print(f"Quantized model accuracy: {nntool_quant_accuracy}")
