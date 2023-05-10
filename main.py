# from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Response, status
from fastapi.responses import FileResponse, StreamingResponse

import os
import sys
import cv2
import numpy as np
import io
from datetime import datetime

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from recognition import Recognition
from detection import Detector
import settings
import utils
from db.powerpostgre import PowerPost
import time
import faiss as fs

app = FastAPI()

def create_triton_client(server, protocol, verbose, async_set, cpu_server):
    triton_client = None
    if settings.use_cpu:
        try:
            if protocol == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(url=cpu_server, verbose=verbose)
                triton_client.is_server_live()
            else:
                # Specify large enough concurrency to handle the number of requests.
                concurrency = 20 if async_set else 1
                triton_client = httpclient.InferenceServerClient(url=cpu_server, verbose=verbose, concurrency=concurrency)
            print('Connected to CPU Model Server')
        except Exception as e:
            print("CPU client creation failed: " + str(e))
            # sys.exit(1)
    else:
        try:
            if protocol == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(url=server, verbose=verbose)
                triton_client.is_server_live()
            else:
                # Specify large enough concurrency to handle the number of requests.
                concurrency = 20 if async_set else 1
                triton_client = httpclient.InferenceServerClient(url=server, verbose=verbose, concurrency=concurrency)
            print('Connected to GPU Model Server')
        except Exception as e:
            print("GPU client creation failed: " + str(e))
            triton_client = None
            # sys.exit(1)
    return triton_client

triton_client = create_triton_client(settings.TRITON_SERVER_SETTINGS[0], settings.TRITON_SERVER_SETTINGS[1], settings.TRITON_SERVER_SETTINGS[2], settings.TRITON_SERVER_SETTINGS[3], settings.TRITON_SERVER_SETTINGS[4])
detector = Detector(triton_client, settings.use_cpu, settings.DETECTOR_SETTINGS[0], settings.DETECTOR_SETTINGS[1], settings.DETECTOR_SETTINGS[2], settings.DETECTOR_SETTINGS[3], settings.DETECTOR_SETTINGS[4], settings.DETECTOR_SETTINGS[5], settings.DETECTOR_SETTINGS[6])
recognizer = Recognition(triton_client, settings.use_cpu, settings.RECOGNITION_SETTINGS[0], settings.RECOGNITION_SETTINGS[1], settings.RECOGNITION_SETTINGS[2], settings.RECOGNITION_SETTINGS[3], settings.RECOGNITION_SETTINGS[4])
if settings.use_postgres:
    db_worker = PowerPost(settings.PG_CONNECTION[0], settings.PG_CONNECTION[1], settings.PG_CONNECTION[2], settings.PG_CONNECTION[3], settings.PG_CONNECTION[4])
else:
    faiss_index = fs.read_index(settings.FAISS_INDEX_FILE, fs.IO_FLAG_ONDISK_SAME_DIR)

@app.post("/detector/detect", status_code=200)
async def detect_from_photo(response: Response, file: UploadFile = File(...)):
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
            faces = None
            landmarks = None
            # using either gpu or cpu to get feature - the use_cpu is in settings.py
            if settings.use_cpu:
                faces, landmarks = detector.cpu_detect(image, name, settings.DETECTION_THRESHOLD)
            else:
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
async def get_aligned_photo(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
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
        feature = None
        # using either gpu or cpu to get feature - the use_cpu is in settings.py
        if settings.use_cpu:
            feature = recognizer.cpu_get_feature(img, unique_id+'_'+img_name)
        else:
            feature = recognizer.get_feature(img, unique_id+'_'+img_name, 0)

        if settings.use_postgres:
            # get top one from postgres database of people
            db_result = db_worker.get_top_one_from_face_db(feature)
        else:
            if faiss_index.ntotal > 0:
                distances, indexes = db_worker.search_from_gbdfl_faiss_top_n(faiss_index, feature, 1)
            else:
                return {'result': 'error', 'message': 'FAISS index is empty.'}
        
            if indexes is not None:
                result_dict = dict()
                ids = tuple(list(map(str,indexes[0])))
                # ids = str(list(indexes[0]))[1:-1]
                print("IDs", ids)
                with_zeros = []
                str_ids = list(map(str, indexes[0]))
                for i in str_ids:
                    while len(i) < 9:
                        i = "0" + i
                    with_zeros.append(i)
                print('ZEROs ADDED:', with_zeros)
                from_ud_gr = db_worker.get_blob_info_from_database(tuple(with_zeros))
                print('FROM DATABASE:', from_ud_gr)
                if from_ud_gr is not None:
                    scores_val = dict(zip(list(with_zeros),list(distances[0])))
                    print('DICTIONARY:', scores_val)
                    for i in range(len(from_ud_gr)):
                        dist = scores_val[from_ud_gr[i][0]]
                        ud_code = from_ud_gr[i][0]
                        gr_code = from_ud_gr[i][1]
                        surname = from_ud_gr[i][2]
                        firstname = from_ud_gr[i][3]
                        if from_ud_gr[i][4] is None:
                            secondname = ''
                        else:
                            secondname = from_ud_gr[i][4]
                        fio = surname +' '+ firstname +' '+secondname
                        result_dict[str(from_ud_gr[i][0])] = {
                                                                'result': 'success',
                                                                'distance': round(dist*100, 2),
                                                                'iin': gr_code,
                                                                'ud_number': ud_code,
                                                                'surname': surname,
                                                                'firstname': firstname,
                                                                'secondname': secondname
                                                                }
                    print('RESULT DICT:', result_dict)
                    return result_dict
            else:
                return {'result': 'error', 'message': 'ud_gr is empty'}

        if len(db_result) > 0:
            ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
            if ids is not None:
                l_name = db_worker.search_from_persons(ids)
                return {
                        'result': 'success',
                        'message': {
                                    'id': ids,
                                    'name': l_name,
                                    'similarity': round(distances * 100, 2)
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
        # using either gpu or cpu to get feature - the use_cpu is in settings.py
        if settings.use_cpu:
            feature = recognizer.cpu_get_feature(img, unique_id+'_'+img_name)
        else:
            feature = recognizer.get_feature(img, unique_id+'_'+img_name, 0)

        db_result = db_worker.one_to_one(feature, person_id)
        if len(db_result) > 0:
            ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
            if ids is not None:
                l_name = db_worker.search_from_persons(ids)
                return {
                        'result': 'success',
                        'message': 'Person matches with person in Database',
                        'id': ids,
                        'name': l_name,
                        'similarity': round(distances * 100, 2)
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
        feature = None
        # using either gpu or cpu to get feature - the use_cpu is in settings.py
        if settings.use_cpu:
            feature = recognizer.cpu_get_feature(img, unique_id)
        else:
            feature = recognizer.get_feature(img, unique_id, 0)
        # getting database results so not to add existing person again
        db_result = db_worker.get_top_one_from_face_db(feature)
        if len(db_result) > 0:
            ids, distances = utils.calculate_cosine_distance(db_result, feature, settings.RECOGNITION_THRESHOLD)
            if ids is not None:
                l_name = db_worker.search_from_persons(ids)
                response.status_code = status.HTTP_409_CONFLICT
                return {
                        'result': 'error',
                        'message': 'Person is already registered in database.',
                        'name': l_name,
                        'similarity': round(distances * 100, 2)
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


@app.post("/detector/compare", status_code=200)
async def compare_two_photos(response: Response, file_1: UploadFile = File(...), file_2: UploadFile = File(...)):
    # check to see if an image was uploaded
    if file_1 is not None and file_2 is not None:
        feature_list = []
        for file in [file_1, file_2]:
            # grab the uploaded image
            data = await file.read()
            unique_id = str(round(time.time() * 1000000))
            name = file.filename
            image = np.asarray(bytearray(data), dtype="uint8")
            try:
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except:
                return {'result': 'error', 'message': 'One photo is broken or empty'}

            if settings.use_cpu:
                faces, landmarks = detector.cpu_detect(image, name, settings.DETECTION_THRESHOLD)
            else:
                faces, landmarks = detector.detect(image, name, 0, settings.DETECTION_THRESHOLD)

            if faces.shape[0] > 0:
                for i in range(faces.shape[0]):
                    aligned = utils.get_alignment(faces[i], landmarks[i], image)
                    if aligned is not None:
                        # Get 512-d embedding from aligned image
                        if settings.use_cpu:
                            feature = recognizer.cpu_get_feature(aligned, unique_id)
                        else:
                            feature = recognizer.get_feature(aligned, unique_id, 0)
                        feature_list.append(feature)
                    else:
                        return {'result': 'error', 'message': 'Face not detected or sharp angle'}
            else:
                # There are no faces or no faces that we can detect
                return {'result': 'error', 'message': 'No faces or no faces that we can detect '}
        cosine_dist = np.dot(feature_list[0], feature_list[1])
        if int(cosine_dist*100) > 0:
            similarity = cosine_dist
        else:
            similarity = 0
        return {'result': 'success', 'message': round(similarity * 100, 2)}
    else:
        return {'result': 'error', 'message': 'No photo provided'}


@app.post("/detector/detect_and_draw", status_code=200)
async def detect_and_draw(response: Response, file: UploadFile = File(...)):
    # check to see if an image was uploaded
    if file is not None:
        # grab the uploaded image
        data = await file.read()
        name = file.filename
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image.shape[0] == image.shape[1] == 112:
            res, unique_id, img = utils.process_and_draw_rectangles(image, [0], [0])
            face_list = [i for i in range(len(res))]
            return {'result': 'success', 'unique_id': unique_id, 'faces': 1, 'filetype': file.content_type, 'size': image.shape}
        else:
            if settings.use_cpu:
                faces, landmarks = detector.cpu_detect(image, name, settings.DETECTION_THRESHOLD)
            else:
                faces, landmarks = detector.detect(image, name, 0, settings.DETECTION_THRESHOLD)

            if faces.shape[0] == 1:
                res, unique_id, img = utils.process_and_draw_rectangles(image, faces, landmarks)
                if len(res) > 0:
                    # face_list = [i for i in range(len(res))]
                    # return {'result': 'success', 'unique_id': unique_id, 'faces': face_list, 'filetype': file.content_type, 'size': image.shape}
                    # Encode processed image back to bytes
                    print(img.shape)
                    is_success, buffer = cv2.imencode(".jpg", img)
                    io_buf = io.BytesIO(buffer)
                    # print(type(buffer))
                    # return Response(content=buffer.tobytes(), media_type="image/jpeg")
                    return FileResponse(unique_id)
                else:
                    response.status_code = status.HTTP_412_PRECONDITION_FAILED
                    return {'result': 'error', 'message': 'no_faces', 'amount': int(faces.shape[0])}
            elif faces.shape[0] > 1:
                response.status_code = status.HTTP_412_PRECONDITION_FAILED
                return {'result': 'error', 'message': 'more_than_one_face', 'amount': int(faces.shape[0])}
            else:
                # There are no faces or no faces that we can detect
                response.status_code = status.HTTP_412_PRECONDITION_FAILED
                return {'result': 'error', 'message': 'no_faces', 'amount': int(faces.shape[0])}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {'result': 'error', 'message': 'No photo provided. Please, check that you are sending correct file.'}