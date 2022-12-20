import os
from pathlib import Path

ip = '127.0.0.1'
http_port = '30020'
grpc_port = '30021'
protocol = 'grpc'

det_model = 'detect'
rec_model = 'recognize'
image_size = '900,900'
im_size=[int(image_size.split(',')[0]), int(image_size.split(',')[1])]

DETECTION_THRESHOLD = 0.95

TRITON_SERVER_SETTINGS = [ip + ':' + grpc_port, protocol, False, True]

DETECTOR_SETTINGS = [det_model, '', 1, protocol, im_size, True, True]
RECOGNITION_SETTINGS = [rec_model, '', 1, protocol, True, True]

CROPS_FOLDER = '/crops'
