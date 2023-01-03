import numpy as np
import cv2
import os
import uuid
from datetime import datetime

from align_faces import align_img
import settings

def calculate_cosine_distance(db_result, vector, threshold):
    distance = float(threshold)
    idx = None
    dist = None
    try:
        for row in db_result:
            vec = np.fromstring(row[1][1:-1], dtype=float, sep=',')
            dist = np.dot(vec,vector)
            if dist > distance:
                idx = row[0]
    except Exception as error:
        print('Error: ' + str(error))
        return idx, dist


def process_faces(img, faces, landmarks):
    face_count = 0
    result = []

    todays_folder = os.path.join(settings.CROPS_FOLDER, datetime.now().strftime("%Y%m%d"))
    # print(todays_folder)
    if not os.path.exists(todays_folder):
        os.makedirs(todays_folder)

    # img_name = str(round(time.time() * 1000000))
    img_name = str(uuid.uuid4())
    new_img_folder = os.path.join(todays_folder, img_name)
    if not os.path.exists(new_img_folder):
        os.makedirs(new_img_folder)

    # if size of an image is 112 then it is already cropped and aligned
    if img.shape[0] == 112:
        cv2.imwrite(new_img_folder+'/crop_'+'0.jpg', img)
        cv2.imwrite(new_img_folder+'/align_'+'0.jpg', img)
        # print('Aligned image saved:', new_img_folder+'/align_'+'0.jpg', img.shape)
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
                landmark5 = landmarks[i].astype(np.int)
                aligned = align_img(img, landmark5)

                # save crop and aligned image
                cv2.imwrite(new_img_folder+'/crop_'+str(i)+'.jpg', cropped_img)
                cv2.imwrite(new_img_folder+'/align_'+str(i)+'.jpg', aligned)
                # print('Align saved:', new_img_folder+'/align_'+str(i)+'.jpg', aligned.shape)
                result.append(face_count)
                face_count += 1
            else:
                pass
                # print('Face is too small or modified or in sharp angle')
    # save original image
    if face_count > 0:
        cv2.imwrite(new_img_folder+'/original.jpg', img)
    return result, img_name