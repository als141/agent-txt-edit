#!/usr/bin/env python
"""
NotionMCP Light

Notion APIを使用してMarkdownファイルとNotionページを同期するMCPサーバー
"""

import os
import sys
import argparse
from dotenv import load_dotenv

from .mcp_server import MCPServer


def main():
    """
    メイン関数

    コマンドライン引数を解析し、MCPサーバーを起動します。
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description="NotionMCP Light - Notion APIを使用してMarkdownファイルとNotionページを同期するMCPサーバー"
    )
    parser.add_argument("--token", help="Notion API Token（指定されない場合は環境変数から取得）")
    args = parser.parse_args()

    # 環境変数の読み込み
    load_dotenv()

    # トークンの取得
    token = args.token or os.getenv("NOTION_TOKEN")

    if not token:
        print(
            "エラー: Notion API Tokenが指定されていません。--tokenオプションまたは環境変数NOTION_TOKENを設定してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # MCPサーバーの起動
        server = MCPServer(token)
        server.start()

    except KeyboardInterrupt:
        print("サーバーを終了します。", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
