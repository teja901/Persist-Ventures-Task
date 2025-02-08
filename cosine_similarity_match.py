import asyncio
from fastapi import FastAPI,Form,UploadFile,File
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import concurrent.futures
import tempfile
import shutil

process_pool = concurrent.futures.ProcessPoolExecutor()


app = FastAPI()


model = SentenceTransformer("all-MiniLM-L6-v2")


dimension = 384
index = faiss.IndexFlatL2(dimension)
user_vectors = {}

@app.post("/add_user/")
async def add_user(user_id: str = Form(...), interestsFile: UploadFile = File(...)):
    # interests=interests.strip('[]').split(',')
    interests=await transcribe_audio(interestsFile)
    print(interests)
    vector = model.encode(interests)
    user_vectors[user_id] = vector
    index.add(np.array([vector]))
    # print(user_vectors)
    return {"message": "User added"}

@app.get("/find_matches/{user_id}")
async def find_matches(user_id: str, threshold: float = 1.0):  
    if user_id not in user_vectors:
        return {"message": "User not found"}
    
    user_vector = np.array([user_vectors[user_id]])
    k = len(user_vectors)  
    distances, indices = index.search(user_vector, k)  

    matched_users = []
    for i in range(1, k):  
        matched_user_id = list(user_vectors.keys())[indices[0][i]]
        if distances[0][i] <= threshold:  
            matched_users.append(matched_user_id)

    return {"matched_users": matched_users}

recognizer = sr.Recognizer()

async def convert_audio_to_text(file_path):
    if file_path.endswith(".mp3"):
        AudioSegment.from_mp3(file_path).export(file_path := file_path.replace(".mp3", ".wav"), format="wav")

    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except (sr.UnknownValueError, sr.RequestError):
        return "Error in transcription"

async def transcribe_audio(file):
    if file.filename.split(".")[-1].lower() not in ["mp3", "wav"]:
        return {"error": "Only MP3 or WAV files are supported"}
    

    with tempfile.NamedTemporaryFile(delete=False,suffix=f".{file.filename.split('.')[-1]}") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        return await convert_audio_to_text(temp_file.name)
    # return {"transcription": text, "words": text.split()}