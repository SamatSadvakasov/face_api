# from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Response, status
from fastapi.responses import FileResponse

import os
import cv2
import numpy as np
from datetime import datetime
import time

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from recognition import Recognition
from detection import Detector
import settings
import utils
from db.powerpostgre import PowerPost

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
db_worker = PowerPost(settings.PG_CONNECTION[0], settings.PG_CONNECTION[1], settings.PG_CONNECTION[2], settings.PG_CONNECTION[3], settings.PG_CONNECTION[4])


@app.post("/detector/detect", status_code=200)
async def detect_from_photo(response: Response, file: UploadFile = File(...), unique_id: str = Form(...)):
    # check to see if an image was uploaded
    if file is not None:
        # grab the uploaded image
        data = await file.read()
        name = file.filename
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image.shape[0] == image.shape[1] == 112:
            res, unique_id = utils.process_faces(image, [0], [0])
            face_list = [i for i in range(len(res))]
            return {'result': 'success', 'unique_id': unique_id, 'faces': 1, 'filetype': file.content_type, 'size': image.shape}
        else:
            # [os.remove(settings.CROPS_FOLDER + f) for f in os.listdir(settings.CROPS_FOLDER)]
            faces, landmarks = detector.detect(image, name, 0, settings.DETECTION_THRESHOLD)
            if faces.shape[0] == 1:
                res, unique_id = utils.process_faces(image, faces, landmarks)
                if len(res) > 0:
                    face_list = [i for i in range(len(res))]
                    return {'result': 'success', 'unique_id': unique_id, 'faces': face_list, 'filetype': file.content_type, 'size': image.shape}
                else:
                    response.status_code = status.HTTP_412_PRECONDITION_FAILED
                    return {'result': 'no_faces', 'amount': int(faces.shape[0])}
            elif faces.shape[0] > 1:
                response.status_code = status.HTTP_412_PRECONDITION_FAILED
                return {'result': 'more_than_one_face', 'amount': int(faces.shape[0])}
            else:
                # There are no faces or no faces that we can detect
                response.status_code = status.HTTP_412_PRECONDITION_FAILED
                return {'result': 'no_faces', 'amount': int(faces.shape[0])}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message': 'No photo provided. Please, check that you are sending correct file.'}


@app.get("/detector/get_aligned", status_code=200)
async def get_photo_align(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message':'No such file'}


@app.post("/recognition/get_photo_metadata", status_code=200)
async def get_photo_metadata(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
    img_name = 'align_'+face_id
    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, img_name+'.jpg')
    if os.path.exists(file_path):
        img = cv2.imread(file_path)
        feature = recognizer.get_feature(img, unique_id+'_'+img_name, 0)

        db_result = db_worker.search_from_face_db(feature)
        if len(db_result) > 0:
            ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
            l_name = db_worker.search_from_persons(ids)
            return {
                    'result': 'success',
                    'message': {
                                'id': ids,
                                'name': l_name,
                                'similarity': round(distances, 2)
                            }   
                }
        else:
            response.status_code = status.HTTP_409_CONFLICT
            return {'result': 'error', 'message': 'No IDs found'}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message': 'No such file. Please, check unique_id, face_id or date.'}


@app.post("/recognition/check_person", status_code=200)
async def check_person(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...), person_id: str = Form(...)):
    img_name = 'align_'+face_id
    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, img_name+'.jpg')
    if os.path.exists(file_path):
        img = cv2.imread(file_path)
        feature = recognizer.get_feature(img, unique_id+'_'+img_name, 0)

        db_result = db_worker.one_to_one(feature, person_id)
        if len(db_result) > 0:
            print(type(db_result[0]))
            ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
            l_name = db_worker.search_from_persons(ids)
            return {
                    'result': 'success',
                    'message': 'Person matches with person in Database',
                    'id': ids,
                    'name': l_name,
                    'similarity': round(distances, 2)
                    }
        else:
            response.status_code = status.HTTP_409_CONFLICT
            return {'result': 'error', 'message': 'No IDs found. Either ID you entered is invalid or person does not exist in database.'}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message': 'No such file. Please, check unique_id, face_id or date.'}


@app.post("/database/add_person_to_face_db")
async def add_person_to_face_db(response: Response, 
                                date: str = Form(...), unique_id: str = Form(...), 
                                face_id: str = Form(...), person_name: str = Form(...), 
                                person_surname: str = Form(...), person_secondname: str = Form(...), 
                                group_id: str = Form(...), person_iin: str = Form(...)):
    create_time = datetime.now()
    # getting cropped images from folder - uncomment following line if you want to upload image into your database table
    # db_image = open(os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg'), 'rb').read()
    # getting cropped and aligned image needed to obtain embeddings
    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'align_'+face_id+'.jpg')
    if os.path.exists(file_path):
        img = cv2.imread(file_path)
        feature = recognizer.get_feature(img, unique_id+'_align_'+face_id, 0)

        db_result = db_worker.search_from_face_db(feature)
        if len(db_result) > 0:
            ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
            l_name = db_worker.search_from_persons(ids)
            response.status_code = status.HTTP_409_CONFLICT
            return {
                    'result': 'error',
                    'message': 'Person is already registered in database.',
                    'name': l_name,
                    'similarity': round(distances, 2)
                    }

        result = db_worker.insert_new_person(unique_id, feature, person_name, person_surname, person_secondname, create_time, group_id, person_iin)
        if result:
            return {'result': 'success', 'message': 'Successfully inserted new person.', 'name': person_name, 'unique_id': unique_id}
        else:
            response.status_code = status.HTTP_304_NOT_MODIFIED
            return {'result': 'error', 'message': 'Failed to insert vector to one or more tables.', 'name': person_name, '` unique_id': unique_id}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message': 'No such file. Please, check unique_id, face_id or date.'}
