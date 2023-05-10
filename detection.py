import os, sys
import cv2
import numpy as np
import argparse
import itertools
from functools import partial

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

from postprocessing import *

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


def parse_model_http_detection(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    output_metadata = model_metadata['outputs']

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']


    return (max_batch_size, input_metadata['name'], output_metadata, input_metadata['datatype'])


def parse_model_grpc_detection(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs

    return (model_config.max_batch_size, input_metadata.name, output_metadata, input_metadata.datatype)

def cpu_parse_model_grpc_detection(model_metadata):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs

    return (1, input_metadata.name, output_metadata, input_metadata.datatype)


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


class Detector:
    def __init__(self, triton_client, use_cpu, model_name, model_version, batch_size, protocol, scales, async_set=False, streaming=False):
        self.triton_client = triton_client
        self.scales = scales
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_version = model_version
        self.async_set = async_set
        self.streaming = streaming
        self.protocol = protocol
        """------------------------DETECTION------------------------"""
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self.detection_model_metadata = None
        self.use_cpu = use_cpu
        try:
            self.detection_model_metadata = self.triton_client.get_model_metadata(
                                        model_name=model_name, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)
        try:
            if self.use_cpu:
                self.detection_model_config = None
            else:
                self.detection_model_config = self.triton_client.get_model_config(
                                    model_name=model_name, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)
        self.max_batch_size = None
        self.input_name = None
        self.output_name = None
        self.dtype = None
        if self.protocol == "grpc":
            if self.use_cpu:
                self.max_batch_size, self.input_name, self.output_metadata, self.dtype = cpu_parse_model_grpc_detection(self.detection_model_metadata)
            else:
                self.max_batch_size, self.input_name, self.output_metadata, self.dtype = parse_model_grpc_detection(self.detection_model_metadata, self.detection_model_config.config)
        else:
            self.max_batch_size, self.input_name, self.output_metadata, self.dtype = parse_model_http_detection(self.detection_model_metadata, self.detection_model_config)
        """------------------------DETECTION------------------------"""

    # Callback function used for async_stream_infer()
    def completion_callback(self, user_data, result, error):
        # passing error raise and handling out
        user_data._completed_requests.put((result, error))


    def frame_preprocess(self, img, scales, unique_id):
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        
        #print('im_scale:',scales)
        scales = [im_scale,im_scale]

        resized_img = None
        if im_scale!=1.0:
            resized_img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)        
        else:
            resized_img = img.copy()

        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        dim_changed = np.transpose(resized_img, (2,0,1)) #HWC->CHW

        input_blob = np.expand_dims(dim_changed, axis=0).astype(np.float32)

        images_data = []
        filenames = []
        filename = unique_id
        images_data.append(input_blob)
        filenames.append(filename)

        # print('cpu:', self.use_cpu)
        if self.use_cpu:
            return input_blob, scales, im_shape

        return images_data, scales, filenames, im_shape


    def requestGenerator(self, batched_image_data, input_name, output_metadata, dtype):
        # Set the input data
        inputs = []
        if self.protocol.lower() == "grpc":
            inputs.append(
                grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
            inputs[0].set_data_from_numpy(batched_image_data)
        else:
            inputs.append(httpclient.InferInput(input_name, batched_image_data.shape, dtype))
            inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        output_names = [ output.name if self.protocol.lower() == "grpc"
                            else output['name'] for output in output_metadata ]

        outputs = []
        for output_name in output_names:
            if self.protocol.lower() == "grpc":
                outputs.append(grpcclient.InferRequestedOutput(output_name))
            else:
                outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))

        yield inputs, outputs, self.model_name, self.model_version, output_names


    # response, output_names, threshold, im_scale, scales,  input_filenames, max_batch_size
    def triton_postprocessing(self, results, output_names, threshold, im_scale, scales):
        """
        Post-process results to show classifications.
        """
        # print('results:', results)
        output_dict = {}
        outputs = []
        for output_name in output_names:
            # print('output name:', output_name)
            # print('len(results.as_numpy(output_name)):', len(results.as_numpy(output_name)))
            outputs.append(results.as_numpy(output_name))

        faces, landmarks = postprocess(outputs, threshold, 0, im_scale, scales)

        return faces, landmarks


    def detect(self, img, unique_id, sent_count, detection_threshold):
        image_data, im_scale, filenames, im_shape = self.frame_preprocess(img, self.scales, unique_id)

        requests = []
        responses = []
        result_filenames = []
        request_ids = []
        image_idx = 0
        last_request = False
        user_data = UserData()

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []        
        # print('IMAGE DATA:', len(image_data))
        if self.streaming:
            self.triton_client.start_stream(partial(self.completion_callback, user_data))

        while not last_request:
            input_filenames = []
            repeated_image_data = []

            for idx in range(self.batch_size):
                input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            if self.max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]

            # Send request
            try:
                for inputs, outputs, model_name, model_version, output_names in self.requestGenerator(batched_image_data, self.input_name, self.output_metadata, self.dtype):
                    sent_count += 1
                    if self.streaming:
                        self.triton_client.async_stream_infer(self.model_name, inputs, request_id=str(sent_count), model_version=self.model_version, outputs=outputs)
                    elif self.async_set:
                        if self.protocol.lower() == "grpc":
                            self.triton_client.async_infer(self.model_name, inputs, partial(self.completion_callback, user_data), request_id=str(sent_count), model_version=self.model_version, outputs=outputs)
                        else:
                            async_requests.append(self.triton_client.async_infer(self.model_name, inputs, request_id=str(sent_count), model_version=self.model_version, outputs=outputs))
                    else:
                        responses.append(self.triton_client.infer(self.model_name, inputs, request_id=str(sent_count), model_version=self.model_version, outputs=outputs))
                        result_filenames.append(input_filenames)  #added filename
            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if self.streaming:
                    self.triton_client.stop_stream()
                sys.exit(1)        
        if self.streaming:
            self.triton_client.stop_stream()

        # print(len(responses), 'LAST POINT')
        if self.protocol.lower() == "grpc":
            if self.streaming or self.async_set:
                processed_count = 0
                while processed_count < sent_count:
                    (results, error) = user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        print("inference failed: " + str(error))
                        sys.exit(1)
                    responses.append(results)
        else:
            if self.async_set:
                # Collect results from the ongoing async requests for HTTP Async requests.
                for async_request in async_requests:
                    responses.append(async_request.get_result())
        for (response, fname) in itertools.zip_longest(responses, result_filenames):
            faces, landmarks = self.triton_postprocessing(response, output_names, detection_threshold, im_scale, self.scales)
        return faces, landmarks

    
    def cpu_detect(self, img, unique_id, detection_threshold):
        image_data, im_scale, im_shape = self.frame_preprocess(img, self.scales, unique_id)

        model_input_name = self.detection_model_metadata.inputs[0].name
        infer_input = grpcclient.InferInput(model_input_name, image_data.shape, self.detection_model_metadata.inputs[0].datatype)
        infer_input.set_data_from_numpy(image_data)
        results = self.triton_client.infer(self.model_name, [infer_input])

        output_names = ['face_rpn_cls_prob_reshape_stride32',
                        'face_rpn_bbox_pred_stride32',
                        'face_rpn_landmark_pred_stride32',
                        'face_rpn_cls_prob_reshape_stride16',
                        'face_rpn_bbox_pred_stride16',
                        'face_rpn_landmark_pred_stride16',
                        'face_rpn_cls_prob_reshape_stride8',
                        'face_rpn_bbox_pred_stride8',
                        'face_rpn_landmark_pred_stride8']

        return self.triton_postprocessing(results, output_names, detection_threshold, im_scale, self.scales)
