import os
from pathlib import Path

# to use GPU or CPU processing: True - OVMS, False - Triton Inference Server
use_cpu = True
# use postgre or ANNs: True - Postgres, False - FAISS
use_postgres = True
# IP of the host server
ip = '127.0.0.1'
# http port of the GPU model server
gpu_http_port = '30020'
# grpc port of the GPU model server
gpu_grpc_port = '30021'
# http port of the CPU model server
cpu_http_port = '30024'
# grpc port of the CPU model server
cpu_grpc_port = '30023'
# default protocol to use when doing calls to model server
protocol = 'grpc'
# detection model name on model server
det_model = 'detect'
# recognition model name on model server
rec_model = 'recognize'
# default image size for detection
image_size = '900,900'
im_size=[int(image_size.split(',')[0]), int(image_size.split(',')[1])]
# minimum head size in pixels to detect
min_head_size = 40
# detection and recognition thresholds between 0-1
DETECTION_THRESHOLD = 0.95
RECOGNITION_THRESHOLD = 0.7
# Triton Inference Server settings
TRITON_SERVER_SETTINGS = [ip + ':' + gpu_grpc_port, protocol, False, True, ip + ':' + cpu_grpc_port]
# detection and recognition models' settings
DETECTOR_SETTINGS = [det_model, '', 1, protocol, im_size, True, True]
RECOGNITION_SETTINGS = [rec_model, '', 1, protocol, True, True]
# folder to save cropped heads
CROPS_FOLDER = '/crops'
# only for test purposes
TEST_FILES_DIR = '/app/test_files'
# postgres server settings
pg_server = '127.0.0.1'                             # os.environ['FASTAPI_PG_SERVER'] #10.150.34.13                   #Postgresdb server ip address
pg_port = 30005                                    # os.environ['FASTAPI_PG_PORT'] #5444                               #Postgresdb server default port
pg_db = 'face_db'                                   # os.environ['FASTAPI_PG_DB'] #face_reco                              #Postgresdb database name
pg_username = 'face_reco_admin'                     # os.environ['FASTAPI_PG_USER'] #face_reco_admin                #Postgresdb username
pg_password = 'qwerty123'                           # os.environ['FASTAPI_PG_PASS'] #qwerty123                      #Postgresdb password
# postgres connection settings: host, port, dbname, user, pwd
PG_CONNECTION = [pg_server, pg_port, pg_db, pg_username, pg_password]
# FAISS index settings
FAISS_INDEX_FILE = '/final_index/populated.index'
VECTOR_DIMENSIONS = 512
INDEX_TYPE = 'IVF1,Flat'
FAISS_THRESHOLD = 65
TRAINED_INDEX_PATH = '/trained_index/trained.index'
ALL_INDEXES_PATH = '/storage/indexes'