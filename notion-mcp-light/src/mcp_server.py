"""
MCPサーバーモジュール

Model Context Protocol (MCP)に準拠したサーバーを提供します。
JSON-RPC over stdioを使用してクライアントからのリクエストを処理します。
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.notion_client import NotionClient


class MCPServer:
    """
    Model Context Protocol (MCP)に準拠したサーバークラス

    JSON-RPC over stdioを使用してクライアントからのリクエストを処理します。

    Attributes:
        notion_client: NotionClientのインスタンス
        logger: ロガー
    """

    def __init__(self, token: Optional[str] = None):
        """
        MCPServerのコンストラクタ

        Args:
            token: Notion API Token（指定されない場合は環境変数から取得）
        """
        self.notion_client = NotionClient(token)

        # ロガーの設定
        self.logger = logging.getLogger("mcp_server")
        self.logger.setLevel(logging.INFO)

        # ファイルハンドラの設定
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "mcp_server.log")
        file_handler.setLevel(logging.INFO)

        # フォーマッタの設定
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # ハンドラの追加
        self.logger.addHandler(file_handler)

    def start(self):
        """
        サーバーを起動し、stdioからのリクエストをリッスンします。
        """
        self.logger.info("MCPサーバーを起動しました")

        # サーバー情報を出力
        self._send_response(
            {
                "jsonrpc": "2.0",
                "method": "server/info",
                "params": {
                    "name": "notion-mcp-light",
                    "version": "0.1.0",
                    "description": "Notion MCP Light Server",
                    "tools": self._get_tools(),
                    "resources": self._get_resources(),
                },
            }
        )

        # ツール情報を出力
        self._send_response(
            {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {
                    "tools": self._get_tools(),
                },
            }
        )

        # リクエストをリッスン
        while True:
            try:
                # 標準入力からリクエストを読み込む
                request_line = sys.stdin.readline()
                if not request_line:
                    break

                # リクエストをパース
                request = json.loads(request_line)
                self.logger.info(f"リクエストを受信しました: {request}")

                # リクエストを処理
                self._handle_request(request)

            except json.JSONDecodeError:
                self.logger.error("JSONのパースに失敗しました")
                self._send_error(-32700, "Parse error", None)

            except Exception as e:
                self.logger.error(f"エラーが発生しました: {str(e)}")
                self._send_error(-32603, f"Internal error: {str(e)}", None)

    def _handle_request(self, request: Dict[str, Any]):
        """
        リクエストを処理します。

        Args:
            request: JSONリクエスト
        """
        # リクエストのバリデーション
        if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
            self._send_error(-32600, "Invalid Request", request.get("id"))
            return

        if "method" not in request:
            self._send_error(-32600, "Method not specified", request.get("id"))
            return

        # メソッドの取得
        method = request["method"]
        params = request.get("params", {})
        request_id = request.get("id")

        # メソッドの処理
        if method == "initialize":
            self._handle_initialize(params, request_id)
        elif method == "tools/list":
            self._handle_tools_list(request_id)
        elif method == "tools/call":
            self._handle_tools_call(params, request_id)
        elif method == "uploadMarkdown":
            self._handle_upload_markdown(params, request_id)
        elif method == "downloadMarkdown":
            self._handle_download_markdown(params, request_id)
        else:
            self._send_error(-32601, f"Method not found: {method}", request_id)

    def _handle_initialize(self, params: Dict[str, Any], request_id: Any):
        """
        initializeメソッドを処理します。

        Args:
            params: リクエストパラメータ
            request_id: リクエストID
        """
        # クライアント情報を取得（オプション）
        client_name = params.get("client_name", "unknown")
        client_version = params.get("client_version", "unknown")

        self.logger.info(f"クライアント '{client_name} {client_version}' が接続しました")

        # サーバーの機能を返す
        response = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "notion-mcp-light", "version": "0.1.0", "description": "Notion MCP Light Server"},
            "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False, "subscribe": False}},
        }

        self._send_result(response, request_id)

        # ツール情報を送信
        self._send_response(
            {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {
                    "tools": self._get_tools(),
                },
            }
        )

    def _handle_upload_markdown(self, params: Dict[str, Any], request_id: Any):
        """
        uploadMarkdownメソッドを処理します。

        Args:
            params: リクエストパラメータ
            request_id: リクエストID
        """
        # パラメータのバリデーション
        if "filepath" not in params:
            self._send_error(-32602, "Invalid params: filepath is required", request_id)
            return

        filepath = params["filepath"]
        database_id = params.get("database_id")

        try:
            # Markdownをアップロード
            page_id = self.notion_client.upload_markdown(filepath, database_id)

            # レスポンスを送信
            self._send_result({"page_id": page_id}, request_id)

        except FileNotFoundError:
            self._send_error(-32602, f"File not found: {filepath}", request_id)

        except Exception as e:
            self._send_error(-32603, f"Failed to upload markdown: {str(e)}", request_id)

    def _handle_download_markdown(self, params: Dict[str, Any], request_id: Any):
        """
        downloadMarkdownメソッドを処理します。

        Args:
            params: リクエストパラメータ
            request_id: リクエストID
        """
        # パラメータのバリデーション
        if "page_id" not in params:
            self._send_error(-32602, "Invalid params: page_id is required", request_id)
            return

        if "output_path" not in params:
            self._send_error(-32602, "Invalid params: output_path is required", request_id)
            return

        page_id = params["page_id"]
        output_path = params["output_path"]

        try:
            # ページをダウンロード
            self.notion_client.download_page(page_id, output_path)

            # レスポンスを送信
            self._send_result({"output_path": output_path}, request_id)

        except Exception as e:
            self._send_error(-32603, f"Failed to download markdown: {str(e)}", request_id)

    def _send_result(self, result: Any, request_id: Any):
        """
        成功レスポンスを送信します。

        Args:
            result: レスポンス結果
            request_id: リクエストID
        """
        response = {"jsonrpc": "2.0", "result": result, "id": request_id}

        self._send_response(response)

    def _send_error(self, code: int, message: str, request_id: Any):
        """
        エラーレスポンスを送信します。

        Args:
            code: エラーコード
            message: エラーメッセージ
            request_id: リクエストID
        """
        response = {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": request_id}

        self._send_response(response)

    def _send_response(self, response: Dict[str, Any]):
        """
        レスポンスを標準出力に送信します。

        Args:
            response: レスポンス
        """
        response_json = json.dumps(response)
        print(response_json, flush=True)
        self.logger.info(f"レスポンスを送信しました: {response_json}")

    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        サーバーが提供するツールの一覧を取得します。

        Returns:
            ツールの一覧
        """
        return [
            {
                "name": "uploadMarkdown",
                "description": "Markdownファイルをアップロードし、Notionページとして作成します",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "アップロードするMarkdownファイルのパス"},
                        "database_id": {
                            "type": "string",
                            "description": "アップロード先のデータベースID",
                        },
                        "page_id": {
                            "type": "string",
                            "description": "親ページID（database_idが指定されていない場合に使用）",
                        },
                    },
                    "required": ["filepath"],
                },
            },
            {
                "name": "downloadMarkdown",
                "description": "NotionページをダウンロードしてMarkdownファイルとして保存します",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "page_id": {"type": "string", "description": "ダウンロードするNotionページのID"},
                        "output_path": {"type": "string", "description": "出力先のファイルパス"},
                    },
                    "required": ["page_id", "output_path"],
                },
            },
        ]

    def _handle_tools_call(self, params: Dict[str, Any], request_id: Any):
        """
        tools/callメソッドを処理します。

        Args:
            params: リクエストパラメータ
            request_id: リクエストID
        """
        # パラメータのバリデーション
        if "name" not in params:
            self._send_error(-32602, "Invalid params: name is required", request_id)
            return

        if "arguments" not in params:
            self._send_error(-32602, "Invalid params: arguments is required", request_id)
            return

        tool_name = params["name"]
        arguments = params["arguments"]

        # ツールの処理
        if tool_name == "uploadMarkdown":
            try:
                page_id = self.notion_client.upload_markdown(
                    arguments["filepath"], arguments.get("database_id"), arguments.get("page_id")
                )
                self._send_result(
                    {"content": [{"type": "text", "text": f"Markdownファイルをアップロードしました。ページID: {page_id}"}]},
                    request_id,
                )
            except FileNotFoundError:
                self._send_result(
                    {
                        "content": [{"type": "text", "text": f"ファイルが見つかりません: {arguments.get('filepath')}"}],
                        "isError": True,
                    },
                    request_id,
                )
            except Exception as e:
                self._send_result(
                    {
                        "content": [{"type": "text", "text": f"Markdownファイルのアップロードに失敗しました: {str(e)}"}],
                        "isError": True,
                    },
                    request_id,
                )
        elif tool_name == "downloadMarkdown":
            try:
                self.notion_client.download_page(arguments["page_id"], arguments["output_path"])
                self._send_result(
                    {
                        "content": [
                            {"type": "text", "text": f"Notionページをダウンロードしました。出力先: {arguments['output_path']}"}
                        ]
                    },
                    request_id,
                )
            except Exception as e:
                self._send_result(
                    {
                        "content": [{"type": "text", "text": f"Notionページのダウンロードに失敗しました: {str(e)}"}],
                        "isError": True,
                    },
                    request_id,
                )
        else:
            self._send_result(
                {"content": [{"type": "text", "text": f"ツールが見つかりません: {tool_name}"}], "isError": True}, request_id
            )

    def _handle_tools_list(self, request_id: Any):
        """
        tools/listメソッドを処理します。

        Args:
            request_id: リクエストID
        """
        tools = self._get_tools()
        self._send_result({"tools": tools}, request_id)

    def _get_resources(self) -> List[Dict[str, Any]]:
        """
        サーバーが提供するリソースの一覧を取得します。

        Returns:
            リソースの一覧
        """
        return []
