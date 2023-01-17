import os
import tempfile
import unittest
from unittest.mock import patch

class TestGetPhotoAlign(unittest.TestCase):
    @patch('os.path')
    def test_get_aligned_photo_success(self, mock_path):
        # Arrange
        mock_path.exists.return_value = True
        date = '2022-01-01'
        unique_id = '12345'
        face_id = 'face1'
        expected_file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
        mock_path.join.return_value = expected_file_path
        # Act
        response = await get_aligned_photo(date=date, unique_id=unique_id, face_id=face_id)
        # Assert
        mock_path.join.assert_called_with(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
        self.assertEqual(response.file_path, expected_file_path)
        self.assertEqual(response.status_code, 200)

    @patch('os.path')
    def test_get_aligned_photo_not_found(self, mock_path):
        # Arrange
        mock_path.exists.return_value = False
        date = '2022-01-01'
        unique_id = '12345'
        face_id = 'face1'
        expected_file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
        mock_path.join.return_value = expected_file_path
        # Act
        response = await get_aligned_photo(date=date, unique_id=unique_id, face_id=face_id)
        # Assert
        mock_path.join.assert_called_with(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
        self.assertEqual(response['result'], 'error')
        self.assertEqual(response['message'], 'No such file')

    # generating tests for detect_from_photo    
    import unittest
    from unittest.mock import MagicMock

    class TestDetectFromPhoto(unittest.TestCase):
        def test_detect_from_photo_success(self):
            # create a mock file object with a test image
            mock_file = MagicMock(spec=['read', 'filename', 'content_type'])
            mock_file.read.return_value = b'image_data'
            mock_file.filename = 'test.jpg'
            mock_file.content_type = 'image/jpeg'

            # create a mock response object
            mock_response = MagicMock(spec=['status_code'])
            mock_response.status_code = 200

            # call the function with the mock file and response objects
            result = detect_from_photo(mock_response, file=mock_file)

            # assert the function returned the expected result
            self.assertEqual(result, {'result': 'success', 'unique_id': '', 'faces': 1, 'filetype': 'image/jpeg', 'size': (112, 112, 3)})
            self.assertEqual(mock_response.status_code, 200)

        def test_detect_from_photo_no_photo(self):
            # create a mock response object
            mock_response = MagicMock(spec=['status_code'])
            mock_response.status_code = 200

            # call the function with no file provided
            result = detect_from_photo(mock_response)

            # assert the function returned the expected result
            self.assertEqual(result, {'result': 'error', 'message': 'No photo provided. Please, check that you are sending correct file.'})
            self.assertEqual(mock_response.status_code, 404)

        def test_detect_from_photo_no_faces(self):
            # create a mock file object with a test image
            mock_file = MagicMock(spec=['read', 'filename', 'content_type'])
            mock_file.read.return_value = b'image_data'
            mock_file.filename = 'test.jpg'
            mock_file.content_type = 'image/jpeg'

            # create a mock response object
            mock_response = MagicMock(spec=['status_code'])
            mock_response.status_code = 200

            # call the function with the mock file and response objects
            result = detect_from_photo(mock_response, file=mock_file)

            # assert the function returned the expected result
            self.assertEqual(result, {'result': 'no_faces', 'amount': 0})
            self.assertEqual(mock_response.status_code, 412)


    if __name__ == '__main__':
        unittest.main()

    # generating tests for detect_from_photo

from fastapi.testclient import TestClient

def test_get_aligned_photo_success(client: TestClient):
    # Arrange
    date = '2022-01-01'
    unique_id = '12345'
    face_id = 'face1'
    expected_file_path = os.path.join(settings.CROPS_FOLDER, date, unique_id, 'crop_'+face_id+'.jpg')
    with open(expected_file_path, 'wb') as f:
        f.write(b'test')
    # Act
    response = client.get(f'/get_aligned_photo?date={date}&unique_id={unique_id}&face_id={face_id}')
    # Assert
    assert response.status_code == 200
    assert response.body == b'test'

def test_get_aligned_photo_not_found(client: TestClient):
    # Arrange
    date = '2022-01-01'
    unique_id = '12345'
    face_id = 'face1'
    # Act
    response = client.get(f'/get_aligned_photo?date={date}&unique_id={unique_id}&face_id={face_id}')
    # Assert
    assert response.status_code == 404
    assert response.json() == {'result': 'error', 'message':'No such file'}