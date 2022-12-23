# from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Response, status
from fastapi.responses import FileResponse

import os
import cv2
import numpy as np
from datetime import datetime
import time

from sklearn import preprocessing

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from recognition import Recognition
from detection import Detector
from align_faces import align_img
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


@app.post("/detector/", status_code=200)
async def get_photo_align_large_files(response: Response, file: UploadFile = File(...), unique_id: str = Form(...)):
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
            return {'result': 'success', 'unique_id': unique_id, 'filetype': file.content_type, 'size': image.shape}
        else:
            # [os.remove(settings.CROPS_FOLDER + f) for f in os.listdir(settings.CROPS_FOLDER)]
            faces, landmarks = detector.detect(image, name, 0, settings.DETECTION_THRESHOLD)
            print('faces.shape[0]', faces.shape[0])
            if faces.shape[0] == 1:
                res, unique_id = process_faces(image, faces, landmarks)
                if len(res) > 0:
                    face_list = [i for i in range(len(res))]
                    return {'unique_id': unique_id, 'faces': face_list, 'filetype': file.content_type, 'size': image.shape}
                else:
                    message = {'res' : None}
            elif faces.shape[0] > 1:
                response.status_code = status.HTTP_412_PRECONDITION_FAILED
                return {'result': 'more_than_one_face', 'amount': int(faces.shape[0])}
            else:
                # There are no faces or no faces that we can detect
                response.status_code = status.HTTP_412_PRECONDITION_FAILED
                return {'result': 'no_faces', 'amount': int(faces.shape[0])}
    else:
        return {'result': 'error', 'message': 'No photo provided'}


@app.get("/aligned/", status_code=200)
async def get_photo_align(response: Response, date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...)):
    if unique_id is not None:
        if os.path.exists(settings.CROPS_FOLDER + '/' + date +'/' + unique_id + '/' + 'crop_'+face_id+'.jpg'):
            file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
            return FileResponse(file_path)
        else:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {'error': 'No such file'}
    else:
        response.status_code = status.HTTP_412_PRECONDITION_FAILED
        return {'result': 'Error', 'message': 'please, provide unique_id'}


@app.post("/detector/get_photo_metadata/", status_code=200)
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
    message = {'result': 'Success', 'similarity': ''}
    return message


@app.post("/database/add_person_to_face_db/")
async def add_person_to_face_db(date: str = Form(...), unique_id: str = Form(...), face_id: str = Form(...), person_name: str = Form(...), person_surname: str = Form(...), person_secondname: str = Form(...), group_id: int = Form(...)):
    faiss_index = None
    message = None
    # print(face_id, red_id, red_name, group_id)
    insert_date = datetime.now()
    # print('PHOTO:',os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg'))
    db_image = open(os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg'), 'rb').read()
    i_extension = 'jpg'

    file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'align_'+face_id+'.jpg')
    img = cv2.imread(file_path)
    feature = recognizer.get_feature(img, unique_id+'_align_'+face_id, 0)

    if not os.path.exists(settings.FAISS_INDEX_FILE):
        faiss_index = fs.index_factory(settings.VECTOR_DIMENSIONS, settings.INDEX_TYPE, fs.METRIC_INNER_PRODUCT)
    else:
        faiss_index = fs.read_index(settings.FAISS_INDEX_FILE)
        # request.session.get('fs_index',fs.index_factory(settings.VECTOR_DIMENSIONS, settings.INDEX_TYPE, fs.METRIC_INNER_PRODUCT))
    distances = None
    indexes = None
    if faiss_index.ntotal > 0:
        distances, indexes = db_worker.search_from_blacklist_faiss_top_1(faiss_index, feature, 1, settings.FAISS_THRESHOLD)
        if distances is not None:
            # red_name = db_worker.search_from_application_blacklist(indexes[0])
            return {'message': 'Such person exists', 'name': red_name, 'similarity': round(distances[0]*100, 2)}

    faiss_res = db_worker.insert_into_faiss(faiss_index, red_id, feature)
    fs.write_index(faiss_index, settings.FAISS_INDEX_FILE)
    print('Number of people in faiss index:', faiss_index.ntotal)

    black_res = db_worker.insert_into_blacklist(red_id, feature)
    # app_res = db_worker.insert_into_application_blacklist(red_id, red_name, insert_date, db_image, i_extension, group_id)
    # if faiss_res and black_res and app_res:
    if faiss_res and black_res:
        message = {'res': 'Success', 'name': red_name, 'red_id': red_id, 'number of people in faiss': faiss_index.ntotal}
    else:
        message = {'res': 'Failed to insert feature to one or more tables.', 'faiss-black-appblack:': [faiss_res, black_res]}
    return message


def process_faces(img, faces, landmarks):
    face_count = 0
    result = []

    todays_folder = os.path.join(settings.CROPS_FOLDER, datetime.now().strftime("%Y%m%d"))
    print(todays_folder)
    if not os.path.exists(todays_folder):
        os.makedirs(todays_folder)

    img_name = str(round(time.time() * 1000000))
    new_img_folder = os.path.join(todays_folder, img_name)
    if not os.path.exists(new_img_folder):
        os.makedirs(new_img_folder)

    # if size of an image is 112 then it is already cropped and aligned
    if img.shape[0] == 112:
        # cv2.imwrite(new_img_folder+'/crop_'+'0.jpg', img)
        cv2.imwrite(new_img_folder+'/align_'+'0.jpg', img)
        print('Aligned image saved:', new_img_folder+'/align_'+'0.jpg', img.shape)
        result.append(face_count)
        face_count += 1
    else:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            # Getting the size of head rectangle
            height_y = box[3] - box[1]
            width_x = box[2] - box[0]
            # Calculating cropping area
            if landmarks is not None and height_y > 40:
                '''
                center_y = box[1] + ((box[3] - box[1])/2)
                center_x = box[0] + ((box[2] - box[0])/2)
                rect_y = int(center_y - height_y/2)
                rect_x = int(center_x - width_x/2)
                # Cropping an area
                
                extender = 56
                # height side
                y_start = 0
                im_height = img.shape[0]
                y_end = rect_y + height_y

                if max(0, rect_y-extender) > 0:
                    y_start = rect_y - extender
                if rect_y+height_y+extender > im_height:
                    while y_end <= im_height:
                        y_end = y_end + 1
                else:
                    y_end = rect_y+height_y+extender
                # width side
                x_start = 0
                im_width = img.shape[1]
                x_end = rect_x+width_x
                if max(0, rect_x-extender):
                    x_start = rect_x - extender
                if rect_x+width_x+extender > im_width:
                    while x_end <= im_width:
                        x_end = x_end + 1
                else:
                    x_end = rect_x+width_x+extender

                cropped_img = img[y_start:y_end, x_start:x_end]
                '''
                landmark5 = landmarks[i].astype(np.int)
                aligned = align_img(img, landmark5)

                # save crop and aligned image
                # cv2.imwrite(new_img_folder+'/crop_'+str(i)+'.jpg', cropped_img)
                cv2.imwrite(new_img_folder+'/align_'+str(i)+'.jpg', aligned)
                print('Align saved:', new_img_folder+'/align_'+str(i)+'.jpg', aligned.shape)
                result.append(face_count)
                face_count += 1
            else:
                pass
                # print('Face is too small or modified or in sharp angle')
    # save original image
    if face_count > 0:
        cv2.imwrite(new_img_folder+'/original.jpg', img)
    return result, img_name