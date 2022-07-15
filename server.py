from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from serverModel import predict_sentence

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True,
)

@app.get("/predict")
def predict(input_sentence: str):
  return {'msg': predict_sentence(input_sentence)}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)