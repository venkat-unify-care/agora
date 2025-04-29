import os
import random
import string
import requests
from datetime import datetime
from agora_token_builder import RtcTokenBuilder
from dotenv import load_dotenv

load_dotenv(override=True)
class TokenResponse:
    def __init__(self, token: str, uid: str, channel: str):
        self.token = token
        self.uid = uid
        self.channel = channel

    def __repr__(self):
        return f"TokenResponse(token='{self.token}', uid='{self.uid}', channel='{self.channel}')"

def generate_random_uid() -> int:
    return random.randint(1000000000, 9999999999)  # ensures 6-digit UID

# def generate_channel_name() -> str:
#     timestamp = int(datetime.now().timestamp())
#     random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
#     return f"ai-conversation-{timestamp}-{random_str}"

def generate_token(channel, uid: int = None) -> TokenResponse:
    # app_id = os.environ.get("AGORA_APP_ID")
    # app_cert = os.environ.get("AGORA_APP_CERT")

    # if not app_id or not app_cert:
    #     raise ValueError("AGORA_APP_ID and AGORA_APP_CERTIFICATE must be set in environment variables.")

    # uid = uid if uid else generate_random_uid()
    # channel_name = channel
    # expiration_time = int(datetime.now().timestamp()) + 3600  # valid for 1 hour

    # try:
    #     token = RtcTokenBuilder.buildTokenWithUid(
    #         appId=app_id,
    #         appCertificate=app_cert,
    #         channelName=channel_name,
    #         uid=uid,
    #         role=1,
    #         privilegeExpiredTs=expiration_time
    #     )
    #     return TokenResponse(token=token, uid=str(uid), channel=channel_name)
    # except Exception as e:
    #     raise RuntimeError(f"Failed to generate Agora token: {str(e)}")
    data = {
        "channelId": channel,
        "body": {
            "callerName": "Rufous AI",
        }
    }
    headers={"Content-Type": "application/json"}
    response = requests.post("https://dev.rufous.com/api/notification/joinVideocall?isAnonymous=true", json=data,
        headers=headers,
    )
    return response.json()
