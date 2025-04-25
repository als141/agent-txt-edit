# NotionMCP Light

NotionMCP Lightは、Notion APIを使用してMarkdownファイルとNotionページを同期するModel Context Protocol (MCP)サーバーです。

## 概要

このプロジェクトは、Notionの公式Model Context Protocol (MCP)サーバーが抱える非効率性（Markdownをブロック単位で読み書きし、LLMトークンを消費する点）を解決するために開発されました。トークンを使用せず、API経由で直接MarkdownファイルとNotionのページ／データベースを同期できる非公式のMCPサーバーを提供します。

## 機能

- **Markdown → Notion**
  - H1をページタイトルとして認識
  - Markdownの内容をNotionページまたはデータベースのページとして作成
  - データベースIDを指定可能
  - Notion APIを直接使用（トークン未使用）

- **Notion → Markdown**
  - 指定されたページまたはデータベースのページをMarkdown形式に変換
  - タイトルをH1として出力
  - ブロック構造をMarkdownに変換
  - ファイルに保存

- **MCPサーバー対応**
  - Model Context Protocol（MCP）に準拠
  - CursorやClineなどのAIツールから呼び出し可能なエンドポイントを提供
  - JSON-RPC over stdioベースで動作

## インストール

### 依存関係のインストール

```bash
# uvがインストールされていない場合は先にインストール
# pip install uv

# 依存関係のインストール
uv sync
```

### Notion API Tokenの設定

1. [Notion Developers](https://developers.notion.com/)でアカウントを作成し、APIトークンを取得します。
2. 環境変数に設定するか、`.env`ファイルを作成してトークンを設定します。

```bash
# .envファイルの例
NOTION_TOKEN=your_notion_api_token
```

## 使い方

### MCPサーバーの起動

#### uvを使用する場合（推奨）

```bash
uv run python -m src.main
```

または、トークンを直接指定する場合：

```bash
uv run python -m src.main --token your_notion_api_token
```

#### 通常のPythonを使用する場合

```bash
python -m src.main
```

または、トークンを直接指定する場合：

```bash
python -m src.main --token your_notion_api_token
```

### Cline/Cursorでの設定

Cline/CursorなどのAIツールでNotionMCP Lightを使用するには、`mcp_settings.json`ファイルに以下のような設定を追加します：

```json
"notion-mcp-light": {
  "command": "uv",
  "args": [
    "run",
    "--directory",
    "/path/to/notion-mcp-light",
    "python",
    "-m",
    "src.main"
  ],
  "env": {
    "NOTION_TOKEN": "your_notion_api_token"
  },
  "disabled": false,
  "alwaysAllow": []
}
```

`/path/to/notion-mcp-light`は、NotionMCP Lightのインストールディレクトリに置き換えてください。

## MCPツールの使用方法

NotionMCP Lightは以下のMCPツールを提供します：

### uploadMarkdown

Markdownファイルをアップロードし、Notionページとして作成します。

```json
{
  "jsonrpc": "2.0",
  "method": "uploadMarkdown",
  "params": {
    "filepath": "path/to/markdown.md",
    "database_id": "optional_database_id",
    "page_id": "optional_parent_page_id"
  },
  "id": 1
}
```

### downloadMarkdown

NotionページをダウンロードしてMarkdownファイルとして保存します。

```json
{
  "jsonrpc": "2.0",
  "method": "downloadMarkdown",
  "params": {
    "page_id": "notion_page_id",
    "output_path": "path/to/output.md"
  },
  "id": 2
}
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。
