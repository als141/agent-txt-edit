import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

import base64
from io import BytesIO

with open("/home/als0028/work/shintairiku/agent-txt-edit/seo-gen/input.jpeg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Base64をデコードしてBytesIOバッファを生成
buf = BytesIO(base64.b64decode(image_b64))
buf.name = "input.jpg"  # 拡張子付きでname属性を設定
buf.seek(0)

response = client.images.edit(
    model="grok-2-image-1212",
    image=buf,  # BytesIOインスタンスを渡す
    prompt="この写真をゴッホ風の油彩画にしてください",
    n=1
)
print("Edited image URL:", response.data[0].url)
