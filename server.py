# API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse

import recommender_in_10min

# Create the FastAPI app
app = FastAPI()

origins = [
    "http://better-frontend.s3-website-us-east-1.amazonaws.com",
    "https://what-to-eat-hanam.site",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET", "HEAD"],
    allow_headers=["content-type", "Access-Control-Allow-Origin"],
)

IMAGE_ROOT_DIR = "./image/"

class Text(BaseModel):
    text: str

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/images/{filename}")
def read_item(filename):
    targetFile = IMAGE_ROOT_DIR + filename + ".svg"
    print(f"File Download : {targetFile}")
    return FileResponse(targetFile, media_type='image/svg',filename=filename)

@app.options("/")
def response_options():
    return {"Hallo": "Wollo"}

@app.post('/predict/')
async def create_item(text: Text):
    return recommender_in_10min.predict(text.text)