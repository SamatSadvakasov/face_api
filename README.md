## Face detection and recognition API using Retinaface, QMagFace and FastAPI
The implementation of face detection and recognition service using [Retinaface](https://docs.openvino.ai/latest/omz_models_model_retinaface_resnet50_pytorch.html), [QMagFace](https://arxiv.org/abs/2111.13475) FastAPI and tritonclient.

### Installation
> git clone https://github.com/Talgin/face_api.git
- Change settings.py to point to desired service addresses
- Create folders: aligned

### Running the service
> docker-compose up -d

### Issues
Sometimes you can encounter bbox errors. One solution can be to:
  - Go to rcnn/cython and do (you have to have Cython package installed):
  > python setup.py build_ext --inplace

### CHANGE HISTORY (started this in 30.12.2022)
- 30.12.2022 - created new functionality (insert_new_person) to insert into faces and person to be able to revert changes if one of the inserts fail
- 02.01.2023 - changed SQL scripts to a new (more secure) style, created utils.py

### TO-DO
- [x] Function to add person to postgres database (unique_id, vector)
- [ ] Finish utils.py
- [x] Revert changes to database if insert to some of the tables fails (finish this part!!!!!!!!!)
- [ ] Functionality to compare two photos
- [ ] Finish unit-tests
- [ ] Accept multiple requests at one time - think about it
- [ ] Refine code (object reusability, client creation, database connection, configs)
- [ ] Refine face recognition algo
- [ ] Add Metadata and Docs descriptions according to [FastAPI Docs](https://fastapi.tiangolo.com/tutorial/metadata/)
- [ ] Add scaNN search functionality
- [ ] Add license checking at startup - connect to license server during installation to obtain license file or write to license db mac
- [ ] Add docker images to docker hub and update readme
- [ ] List all licenses in one file
- [ ] Connect with MLOps pipeline
- [ ] Create documentation (dev, user)