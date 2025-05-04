import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)


response = client.images.generate(
    model="grok-2-image-1212",
    prompt="青い空の下で走る赤いスポーツカーを描いてください"
)
url = response.data[0].url
print("生成された画像URL:", url)
