from fastapi import FastAPI, File, UploadFile
import cv2
import uvicorn

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):
    try:
        image = cv2.imdecode(np.frombuffer(file, np.uint8), -1)
        shape = image.shape
        return {"status": "success", "shape": shape}
    except:
        return {"status": "not found"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
