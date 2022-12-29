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

### TO-DO
- [ ] Refince face recognition algo
- [ ] Accept multiple requests at one time - think about it
- [ ] Function to add person to postgres database (unique_id, vector)
- [ ] Functionality to compare two photos
- [ ] Refine code (object reusability, client creation, database connection, configs)
- [ ] Add scaNN search functionality
- [ ] Add docker images to docker hub and update readme
- [ ] Create documentation