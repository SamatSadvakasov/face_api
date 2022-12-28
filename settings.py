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
RECOGNITION_THRESHOLD = 70

TRITON_SERVER_SETTINGS = [ip + ':' + grpc_port, protocol, False, True]

DETECTOR_SETTINGS = [det_model, '', 1, protocol, im_size, True, True]
RECOGNITION_SETTINGS = [rec_model, '', 1, protocol, True, True]

CROPS_FOLDER = '/crops'

TEST_FILES_DIR = '/app/test_files'

pg_server = '127.0.0.1'                             # os.environ['FASTAPI_PG_SERVER'] #10.150.34.13                   #Postgresdb server ip address
pg_port = 30005                                    # os.environ['FASTAPI_PG_PORT'] #5444                               #Postgresdb server default port
pg_db = 'face_db'                                   # os.environ['FASTAPI_PG_DB'] #face_reco                              #Postgresdb database name
pg_username = 'face_reco_admin'                     # os.environ['FASTAPI_PG_USER'] #face_reco_admin                #Postgresdb username
pg_password = 'qwerty123'                           # os.environ['FASTAPI_PG_PASS'] #qwerty123                      #Postgresdb password
# Postgresql connection settings: host, port, dbname, user, pwd
PG_CONNECTION = [pg_server, pg_port, pg_db, pg_username, pg_password]