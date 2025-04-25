"""
Notionクライアントモジュール

NotionのAPIを利用してMarkdownファイルのアップロードとNotionページのダウンロードを行うクラスを提供します。
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from notion_client import Client
from dotenv import load_dotenv

from src.markdown_converter import MarkdownConverter


class NotionClient:
    """
    NotionのAPIを利用してMarkdownファイルのアップロードとNotionページのダウンロードを行うクラス

    Attributes:
        client: Notion APIクライアント
        converter: MarkdownConverterのインスタンス
    """

    def __init__(self, token: Optional[str] = None):
        """
        NotionClientのコンストラクタ

        Args:
            token: Notion API Token（指定されない場合は環境変数から取得）
        """
        # 環境変数からトークンを取得
        load_dotenv()
        self.token = token or os.getenv("NOTION_TOKEN")

        if not self.token:
            raise ValueError("Notion API Tokenが指定されていません。環境変数NOTION_TOKENを設定してください。")

        self.client = Client(auth=self.token)
        self.converter = MarkdownConverter()

    def upload_markdown(self, filepath: str, database_id: Optional[str] = None, page_id: Optional[str] = None) -> str:
        """
        Markdownファイルを読み込み、Notionページとしてアップロードします。

        Args:
            filepath: アップロードするMarkdownファイルのパス
            database_id: アップロード先のデータベースID
            page_id: 親ページID（database_idが指定されていない場合に使用）

        Returns:
            作成されたNotionページのID
        """
        # ファイルを読み込む
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

        with open(file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # Markdownをパースしてブロックに変換
        blocks, title = self.converter.parse_markdown_to_blocks(markdown_content)

        # ページを作成
        if database_id:
            # データベース内にページを作成
            page = self.client.pages.create(
                parent={"database_id": database_id},
                properties={"title": {"title": [{"text": {"content": title}}]}},
                children=blocks,
            )
        elif page_id:
            # 親ページの下に新規ページを作成
            page = self.client.pages.create(
                parent={"page_id": page_id},
                properties={"title": {"title": [{"text": {"content": title}}]}},
                children=blocks,
            )
        else:
            # 親ページIDが指定されていない場合は、エラーを発生させる
            raise ValueError(
                "database_idまたはpage_idを指定してください。ワークスペースに直接ページを作成することはできません。"
            )

        return page["id"]

    def download_page(self, page_id: str, output_path: str) -> None:
        """
        NotionページをダウンロードしてMarkdownファイルとして保存します。

        Args:
            page_id: ダウンロードするNotionページのID
            output_path: 出力先のファイルパス
        """
        # ページ情報を取得
        page = self.client.pages.retrieve(page_id)

        # ページタイトルを取得
        title = ""
        title_property = page["properties"].get("title", {})
        if title_property and "title" in title_property:
            title_items = title_property["title"]
            if title_items:
                title = title_items[0].get("plain_text", "")

        # ブロックを取得
        blocks = []
        has_more = True
        cursor = None

        while has_more:
            if cursor:
                response = self.client.blocks.children.list(block_id=page_id, start_cursor=cursor)
            else:
                response = self.client.blocks.children.list(block_id=page_id)

            blocks.extend(response["results"])
            has_more = response["has_more"]
            cursor = response.get("next_cursor")

        # ブロックをMarkdownに変換
        markdown_content = self.converter.convert_blocks_to_markdown(blocks, title)

        # ファイルに保存
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def get_database_pages(self, database_id: str) -> List[Dict[str, Any]]:
        """
        データベース内のすべてのページを取得します。

        Args:
            database_id: 取得するデータベースのID

        Returns:
            データベース内のページのリスト
        """
        pages = []
        has_more = True
        cursor = None

        while has_more:
            if cursor:
                response = self.client.databases.query(database_id=database_id, start_cursor=cursor)
            else:
                response = self.client.databases.query(database_id=database_id)

            pages.extend(response["results"])
            has_more = response["has_more"]
            cursor = response.get("next_cursor")

        return pages

    def download_database(self, database_id: str, output_dir: str) -> None:
        """
        データベース内のすべてのページをダウンロードしてMarkdownファイルとして保存します。

        Args:
            database_id: ダウンロードするデータベースのID
            output_dir: 出力先のディレクトリパス
        """
        # データベース内のページを取得
        pages = self.get_database_pages(database_id)

        # 出力ディレクトリを作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 各ページをダウンロード
        for page in pages:
            page_id = page["id"]

            # ページタイトルを取得
            title = ""
            title_property = page["properties"].get("title", {})
            if title_property and "title" in title_property:
                title_items = title_property["title"]
                if title_items:
                    title = title_items[0].get("plain_text", "")

            # ファイル名を生成（タイトルがない場合はページIDを使用）
            filename = f"{title or page_id}.md"
            file_path = output_path / filename

            # ページをダウンロード
            self.download_page(page_id, str(file_path))
