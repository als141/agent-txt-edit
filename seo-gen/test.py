import os
from dotenv import load_dotenv
from openai import OpenAI

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得してクライアントを初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
  model="gpt-4o-mini",
  input=[
    {
      "role": "system",
      "content": [
        {
          "type": "input_text",
          "text": "テスト"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "テーマ「節約＆時短！主婦におすすめの芝生メンテナンス術と便利グッズ」が選択されました。キーワード: 芝生 メンテナンス 時短, 芝生 節約 手入れ, 便利グッズ \nガーデニング, 主婦 おすすめ 芝生生成されたリサーチ計画:\nトピック: 節約＆時短！主婦におすすめの芝生メンテナンス術と便利グッズ\n  クエリ 1: 芝生 メンテナンス 時短 効率的な方法 (焦点: \n主婦が忙しい中でも短時間で効率よく芝生の手入れを行うための具体的な方法やテク\nニックを明らかにする)\n  クエリ 2: 芝生 節約 手入れ コスト削減アイデア (焦点: \n芝生のメンテナンスにかかる費用を抑えるための節約術や経済的な手入れ方法を探る\n)\n  クエリ 3: ガーデニング 便利グッズ 主婦 おすすめ 芝生ケア (焦点: \n主婦に特に使いやすく評価の高い芝生の手入れに役立つ便利グッズの種類と特徴を調\n査する)テーマとキーワードに関して検索してください"
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "text"
    }
  },
  reasoning={},
  tools=[
    {
      "type": "web_search_preview",
      "user_location": {
        "type": "approximate",
        "country": "JP"
      },
      "search_context_size": "medium"
    }
  ],
  tool_choice="required",
  temperature=1,
  max_output_tokens=2094,
  top_p=1,
  store=True
)

print(response)