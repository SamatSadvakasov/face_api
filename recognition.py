#!/usr/bin/env python
import numpy as np
import sys
from functools import partial
import cv2
import itertools
from functools import partial

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

from sklearn.preprocessing import normalize

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    return (model_config.max_batch_size, input_metadata.name, output_metadata.name, input_metadata.datatype)


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))
    if len(model_metadata['outputs']) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata['outputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    output_metadata = model_metadata['outputs'][0]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

    if output_metadata['datatype'] != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata['name'] + "' output type is " +
                        output_metadata['datatype'])

    return (max_batch_size, input_metadata['name'], output_metadata['name'], input_metadata['datatype'])


def cpu_parse_model_grpc(model_metadata):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    input_metadata = model_metadata.inputs[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    return (1, input_metadata.name, output_metadata.name, input_metadata.datatype)

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


class Recognition:
    def __init__(self, triton_client, use_cpu, model_name, model_version, batch_size, protocol, async_set=False, streaming=False):
        self.triton_client = triton_client
        self.use_cpu = use_cpu
        self.model_name = model_name
        self.model_version = model_version
        self.batch_size = batch_size
        self.protocol = protocol
        self.async_set = async_set
        self.streaming = streaming        
        self.model_metadata = triton_client.get_model_metadata(
                                model_name=model_name, model_version=model_version)
        if self.use_cpu:
            self.model_config = None
        else:
            self.model_config = triton_client.get_model_config(
                                model_name=model_name, model_version=model_version)
        self.max_batch_size = None
        self.input_name = None
        self.output_name = None
        self.dtype = None
        if self.protocol.lower() == "grpc":
            if self.use_cpu:
                self.max_batch_size, self.input_name, self.output_name, self.dtype = cpu_parse_model_grpc(self.model_metadata)
            else:
                self.max_batch_size, self.input_name, self.output_name, self.dtype = parse_model_grpc(
                                                                                self.model_metadata, self.model_config.config)
        else:
            self.max_batch_size, self.input_name, self.output_name, self.dtype = parse_model_http(
                                                                                self.model_metadata, self.model_config)
                                                                                

    # Callback function used for async_stream_infer()
    def completion_callback(self, user_data, result, error):
        # passing error raise and handling out
        user_data._completed_requests.put((result, error))

    # read the incoming data and preprocess the images into input data according to model requirements
    def preprocess(self, img, filename):
        images_data = []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1))
        input_blob = np.expand_dims(img, axis=0).astype(np.float32) #NCHW
        images_data.append(input_blob)

        if self.use_cpu:
            return input_blob, filename

        return images_data, filename

    # postprocess the output data and print
    def reco_postprocess(self, results, output_name, batch_size, batching):
        output_array = results.as_numpy(output_name)
        if len(output_array) != batch_size:
            raise Exception("expected {} results, got {}".format(
                batch_size, len(output_array)))

        # Include special handling for non-batching models
        if self.use_cpu:
            normalized_feature = normalize(output_array).flatten()
            return normalized_feature

        for results in output_array:
            if not batching:
                results = [results]
            for result in results:
                normalized_feature = normalize(results).flatten()

        return normalized_feature

    def requestGenerator(self, batched_image_data, input_name, output_name, dtype):

        # Set the input data
        inputs = []
        if self.protocol.lower() == "grpc":
            inputs.append(grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
            inputs[0].set_data_from_numpy(batched_image_data)
        else:
            inputs.append(httpclient.InferInput(input_name, batched_image_data.shape, dtype))
            inputs[0].set_data_from_numpy(batched_image_data)

        outputs = []
        if self.protocol.lower() == "grpc":
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        else:
            outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))

        yield inputs, outputs, self.model_name, self.model_version


    def get_feature(self, aligned_image, filename, sent_count):
        image_data, filenames = self.preprocess(aligned_image, filename)

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        responses = []
        result_filenames = []
        image_idx = 0
        last_request = False
        user_data = UserData()

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []

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
                for inputs, outputs, model_name, model_version in self.requestGenerator(batched_image_data, self.input_name, self.output_name, self.dtype):
                    sent_count += 1
                    if self.streaming:
                        self.triton_client.async_stream_infer(self.model_name, inputs, request_id=str(sent_count), model_version=self.model_version, outputs=outputs)
                    elif self.async_set:
                        if self.protocol.lower() == "grpc":
                            self.triton_client.async_infer(self.model_name, inputs, partial(self.completion_callback, user_data), request_id=str(sent_count), model_version=self.model_version, outputs=outputs)
                        else:
                            async_requests.append(
                                self.triton_client.async_infer(self.model_name, inputs, request_id=str(sent_count), model_version=self.model_version, outputs=outputs))
                    else:
                        responses.append(
                            self.triton_client.infer(self.model_name, inputs, request_id=str(sent_count), model_version=self.model_version, outputs=outputs))
                        result_filenames.append(input_filenames)  #added filename
            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if self.streaming:
                    self.triton_client.stop_stream()
                sys.exit(1)

        if self.streaming:
            self.triton_client.stop_stream()

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
            vector = self.reco_postprocess(response, self.output_name, self.batch_size, self.max_batch_size > 0)

        return vector


    def cpu_get_feature(self, aligned_image, filename):
        image_data, filenames = self.preprocess(aligned_image, filename)

        infer_input = grpcclient.InferInput(self.model_metadata.inputs[0].name, image_data.shape, self.model_metadata.inputs[0].datatype)
        infer_input.set_data_from_numpy(image_data)
        results = self.triton_client.infer(self.model_name, [infer_input])

        vector = self.reco_postprocess(results, self.model_metadata.outputs[0].name, 1, 1 > 0)
        
        return vector
