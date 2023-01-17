from fastapi.testclient import TestClient
import settings
import os

from .main import app

client = TestClient(app)

def test_detect_from_photo_aligned_photo():
    test_file = os.path.join(settings.TEST_FILES_DIR, 'Iliya_test.png')
    files = {'file': ('Iliya_test.png', open(test_file, 'rb'))}
    response = client.post('/detector/detect', files=files)
    assert response.status_code == 200
    assert response.json() == {'result': 'success', 'unique_id': 1672033247808148, 'faces': [0], 'filetype': "image/jpeg", 'size': [112,112,3]}


def test_detect_from_photo_fullsize_photo():
    test_file = os.path.join(settings.TEST_FILES_DIR, 'tom_test.jpg')
    files = {'file': ('tom_test.jpg', open(test_file, 'rb'))}
    response = client.post('/detector/detect', files=files)
    assert response.status_code == 200
    assert response.json() == {'result': 'success', 'unique_id': 1672033247808148, 'faces': [0], 'filetype': "image/jpeg", 'size': [1200,1200,3]}


## check_person wrong date

## check_person wrong unique_id

## check_person wrong person_id

## check_person wrong face_id