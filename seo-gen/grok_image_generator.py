import os
import base64
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
import argparse
from dotenv import load_dotenv

load_dotenv()

def setup_api_client():
    """APIクライアントのセットアップを行う関数"""
    # APIキーの設定 (環境変数から読み込む場合)
    api_key = os.environ.get("XAI_API_KEY")
    
    # APIキーが環境変数にない場合は直接入力を求める
    if not api_key:
        api_key = input("xAI API Keyを入力してください: ")
        # 次回のために環境変数に設定することを推奨
        print("次回のために環境変数 XAI_API_KEY にAPIキーを設定することをお勧めします。")
    
    # OpenAI互換のクライアントを作成
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"  # xAI APIのベースURL
    )
    
    return client

def encode_image(image_path):
    """画像をBase64エンコードする関数"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_from_text(client, prompt, num_images=1):
    """テキストプロンプトから画像を生成する関数"""
    try:
        response = client.images.generate(
            model="grok-2-image",  # grok-2-imageモデルを指定
            prompt=prompt,
            n=num_images  # 生成する画像の数
        )
        
        return response
    
    except Exception as e:
        print(f"画像生成中にエラーが発生しました: {e}")
        return None

def generate_image_from_image_and_text(client, image_path, prompt):
    """画像とテキストから新しい画像を生成する関数"""
    try:
        # 画像をBase64エンコード
        base64_image = encode_image(image_path)
        
        # APIリクエストの準備
        # 注意: 現在のAPIドキュメントでは画像入力の正確な形式が明確でないため、
        # これはOpenAIのDALL-E形式に基づく仮の実装です
        response = client.images.edit(
            model="grok-2-image",
            image=base64_image,
            prompt=prompt
        )
        
        return response
    
    except Exception as e:
        print(f"画像生成中にエラーが発生しました: {e}")
        return None

def save_images(response, output_dir="output"):
    """生成された画像を保存する関数"""
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_paths = []
    
    # responseからデータを取り出して画像を保存
    for i, image_data in enumerate(response.data):
        # URLからの場合
        if hasattr(image_data, 'url') and image_data.url:
            # URLから画像をダウンロード
            image_response = requests.get(image_data.url)
            image = Image.open(BytesIO(image_response.content))
            
            # 画像を保存
            file_path = os.path.join(output_dir, f"generated_image_{i}.jpg")
            image.save(file_path)
            saved_paths.append(file_path)
            print(f"画像を保存しました: {file_path}")
        
        # Base64データの場合
        elif hasattr(image_data, 'b64_json') and image_data.b64_json:
            image_bytes = base64.b64decode(image_data.b64_json)
            image = Image.open(BytesIO(image_bytes))
            
            # 画像を保存
            file_path = os.path.join(output_dir, f"generated_image_{i}.jpg")
            image.save(file_path)
            saved_paths.append(file_path)
            print(f"画像を保存しました: {file_path}")
    
    return saved_paths

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="grok-2-imageモデルを使用して画像を生成するツール")
    parser.add_argument("--prompt", type=str, help="画像生成用のテキストプロンプト")
    parser.add_argument("--image", type=str, help="入力画像のパス（オプション）")
    parser.add_argument("--num", type=int, default=1, help="生成する画像の数（テキストのみの場合、デフォルトは1）")
    parser.add_argument("--output", type=str, default="output", help="出力ディレクトリ（デフォルトは'output'）")
    
    args = parser.parse_args()
    
    # プロンプトが指定されていない場合
    if not args.prompt:
        args.prompt = input("画像生成用のプロンプトを入力してください: ")
    
    # APIクライアントのセットアップ
    client = setup_api_client()
    
    # 画像とテキストから画像を生成する場合
    if args.image:
        print(f"入力画像: {args.image}")
        print(f"プロンプト: {args.prompt}")
        response = generate_image_from_image_and_text(client, args.image, args.prompt)
    # テキストだけから画像を生成する場合
    else:
        print(f"プロンプト: {args.prompt}")
        response = generate_image_from_text(client, args.prompt, args.num)
    
    # 応答が有効な場合、画像を保存
    if response:
        saved_paths = save_images(response, args.output)
        if saved_paths:
            print(f"{len(saved_paths)}枚の画像を生成しました。")
    else:
        print("画像の生成に失敗しました。")

if __name__ == "__main__":
    main()