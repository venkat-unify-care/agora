import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import nest_asyncio
from token_gen import generate_token
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Agora server URL (make sure server is running with: python -m realtime_agent.main server)
AGORA_SERVER_URL = "http://localhost:8080"

class InputRequest(BaseModel):
    channel_name: str
    # uid: int
    # token: str

class StopRequest(BaseModel):
    channel_name: str

@app.post('/start-bot')
async def start_bot(input_request: InputRequest):
    if not input_request.channel_name: #or not input_request.token:
        raise HTTPException(status_code=400, detail="channel_name is required")
    
    response=generate_token(input_request.channel_name)

    payload = {
            "uid": response.uid,
            "channel_name": input_request.channel_name,
            "token": response.token
    }
    print("Got the payload!!!")
    print(payload)

    try:
        res = requests.post(f"{AGORA_SERVER_URL}/start_agent", json=payload)
        print('I am in try block')
        res.raise_for_status()
        return {r"message":res.json(),"UID": payload['uid'],"token":payload['token']}
    except requests.exceptions.HTTPError as e:
        print('oops!')
        raise HTTPException(status_code=res.status_code, detail=f"HTTP {res.status_code}: {res.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/stop_agent')
async def stop_bot(stop_request: StopRequest):
    if not stop_request.channel_name:
        raise HTTPException(status_code=400, detail="channel name is required")
    payload={
        "channel_name":stop_request.channel_name
    }
    try:
        res = requests.post(f"{AGORA_SERVER_URL}/stop_agent",json=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=res.status_code, detail=f"HTTP {res.status_code}: {res.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=5000)