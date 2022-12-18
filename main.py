from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse

import os
import cv2
import numpy as np

from sklearn import preprocessing

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from recognition import Recognition
import settings

app = FastAPI()

def create_triton_client(server, protocol, verbose, async_set):
    triton_client = None
    if protocol == "grpc":
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(url=server, verbose=verbose)
    else:
        # Specify large enough concurrency to handle the number of requests.
        concurrency = 20 if async_set else 1
        triton_client = httpclient.InferenceServerClient(url=server, verbose=verbose, concurrency=concurrency)
    return triton_client

triton_client = create_triton_client(settings.TRITON_SERVER_SETTINGS[0], settings.TRITON_SERVER_SETTINGS[1], settings.TRITON_SERVER_SETTINGS[2], settings.TRITON_SERVER_SETTINGS[3])
detector = Detector(triton_client, settings.DETECTOR_SETTINGS[0], settings.DETECTOR_SETTINGS[1], settings.DETECTOR_SETTINGS[2], settings.DETECTOR_SETTINGS[3], settings.DETECTOR_SETTINGS[4], settings.DETECTOR_SETTINGS[5], settings.DETECTOR_SETTINGS[6])
recognizer = Recognition(triton_client, settings.RECOGNITION_SETTINGS[0], settings.RECOGNITION_SETTINGS[1], settings.RECOGNITION_SETTINGS[2], settings.RECOGNITION_SETTINGS[3], settings.RECOGNITION_SETTINGS[4])


@app.post("/detector/")
async def get_photo_align_large_files(file: UploadFile = File(...)):
    message = {"res": False}
    # check to see if an image was uploaded
    if file is not None:
        # grab the uploaded image
        data = await file.read()
        name = file.filename
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print('Image shape:', image.shape)
        if image.shape[0] == image.shape[1] == 112:
            res, unique_id = process_faces(image, [0], [0])
            if len(res) > 0:
                face_list = [i for i in range(len(res))]
                return {'unique_id': unique_id, 'faces': face_list, 'filetype': file.content_type, 'size': image.shape}
            else:
                message = {'res' : None}
        else:
            # [os.remove(settings.CROPS_FOLDER + f) for f in os.listdir(settings.CROPS_FOLDER)]
            faces, landmarks = detector.detect(image, name, 0, settings.DETECTION_THRESHOLD)
            if faces.shape[0] > 0:
                res, unique_id = process_faces(image, faces, landmarks)
                if len(res) > 0:
                    face_list = [i for i in range(len(res))]
                    return {'unique_id': unique_id, 'faces': face_list, 'filetype': file.content_type, 'size': image.shape}
                else:
                    message = {'res' : None}
            else:
                # There are no faces or no faces that we can detect
                message = {"res": False}
    else:
        return {'res': 'No photo provided'}


@app.get("/aligned/")
async def get_photo_align(date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
    if unique_id is not None:
        # -------------------- CHANGE THIS PLACE TO GET CROPPED IMAGE FROM DATE FOLDER --------------------
        if os.path.exists(settings.CROPS_FOLDER + '/' + date +'/' + unique_id + '/' + 'crop_'+face_id+'.jpg'):
            file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
            return FileResponse(file_path)
        else:
            message = {'error': 'No such file'}
            return message


@app.post("/detector/get_photo_metadata/")
async def get_photo_metadata(date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...), top: int = Form(...)):
    img_name = 'align_'+face_id
    if os.path.exists(settings.CROPS_FOLDER+'/'+date+'/'+unique_id+'/'+img_name+'.jpg'):
        file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, img_name+'.jpg')
    else:
        return {'ERROR': 'No such file.'}
    print('file_path:', file_path)
    img = cv2.imread(file_path)
    feature = recognizer.get_feature(img, unique_id+'_'+img_name, 0)
    dct = {unique_id: list(feature)}
    if not os.path.exists(settings.FAISS_INDEX_FILE):
        message = {'res': 'No faiss index created.'}
        return message
    else:
        #faiss_index = fs.read_index(settings.FAISS_INDEX_FILE)
        if faiss_index.ntotal > 0:
            distances, indexes = db_worker.search_from_gbdfl_faiss_top_n(faiss_index, feature, top)
        else:
            message = {'res': 'Faiss index is empty.'}
            return message
    return message
