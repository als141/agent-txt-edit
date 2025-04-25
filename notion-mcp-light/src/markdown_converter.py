"""
Markdownコンバータモジュール

NotionのブロックとMarkdown形式の相互変換を行うクラスを提供します。
"""

from typing import List, Dict, Any
import mistune


class MarkdownConverter:
    """
    MarkdownとNotionブロック間の変換を行うクラス

    MarkdownテキストをパースしてNotionブロック形式に変換する機能と、
    NotionブロックをパースしてMarkdown形式に変換する機能を提供します。
    """

    def __init__(self):
        """
        MarkdownConverterのコンストラクタ
        """
        self.markdown_parser = mistune.create_markdown()

    def parse_markdown_to_blocks(self, md: str) -> List[Dict[str, Any]]:
        """
        Markdownテキストをパースし、Notionブロック形式に変換します。

        Args:
            md: 変換するMarkdownテキスト

        Returns:
            Notionブロック形式のリスト
        """
        # Markdownを行ごとに分割
        lines = md.strip().split("\n")
        blocks = []

        # タイトル（H1）を抽出
        title = None
        content_start_idx = 0

        # H1をタイトルとして扱う
        for i, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                content_start_idx = i + 1
                break

        # タイトルがない場合は空文字を設定
        if title is None:
            title = ""

        # 残りの内容をブロックに変換
        i = content_start_idx
        while i < len(lines):
            line = lines[i]

            # 見出し（H2-H6）
            if line.startswith("## "):
                blocks.append(
                    {
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]},
                    }
                )
            elif line.startswith("### "):
                blocks.append(
                    {
                        "type": "heading_3",
                        "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:].strip()}}]},
                    }
                )
            # 箇条書き
            elif line.startswith("- "):
                blocks.append(
                    {
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]},
                    }
                )
            # 番号付きリスト
            elif line.strip() and line[0].isdigit() and ". " in line:
                content = line.split(". ", 1)[1]
                blocks.append(
                    {
                        "type": "numbered_list_item",
                        "numbered_list_item": {"rich_text": [{"type": "text", "text": {"content": content.strip()}}]},
                    }
                )
            # コードブロック
            elif line.startswith("```"):
                code_lines = []
                language = line[3:].strip()
                i += 1

                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1

                blocks.append(
                    {
                        "type": "code",
                        "code": {
                            "rich_text": [{"type": "text", "text": {"content": "\n".join(code_lines)}}],
                            "language": language if language else "plain text",
                        },
                    }
                )
            # 通常のテキスト（段落）
            elif line.strip():
                blocks.append(
                    {"type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": line.strip()}}]}}
                )

            i += 1

        return blocks, title

    def convert_blocks_to_markdown(self, blocks: List[Dict[str, Any]], title: str = None) -> str:
        """
        Notionブロックをパースし、Markdown形式に変換します。

        Args:
            blocks: 変換するNotionブロックのリスト
            title: ページのタイトル（指定された場合はH1として出力）

        Returns:
            Markdown形式のテキスト
        """
        md_lines = []

        # タイトルがあればH1として追加
        if title:
            md_lines.append(f"# {title}")
            md_lines.append("")  # 空行を追加

        for block in blocks:
            block_type = block.get("type")

            if block_type == "paragraph":
                text_content = self._extract_text_content(block.get("paragraph", {}).get("rich_text", []))
                md_lines.append(text_content)
                md_lines.append("")  # 空行を追加

            elif block_type == "heading_1":
                text_content = self._extract_text_content(block.get("heading_1", {}).get("rich_text", []))
                md_lines.append(f"# {text_content}")
                md_lines.append("")

            elif block_type == "heading_2":
                text_content = self._extract_text_content(block.get("heading_2", {}).get("rich_text", []))
                md_lines.append(f"## {text_content}")
                md_lines.append("")

            elif block_type == "heading_3":
                text_content = self._extract_text_content(block.get("heading_3", {}).get("rich_text", []))
                md_lines.append(f"### {text_content}")
                md_lines.append("")

            elif block_type == "bulleted_list_item":
                text_content = self._extract_text_content(block.get("bulleted_list_item", {}).get("rich_text", []))
                md_lines.append(f"- {text_content}")

            elif block_type == "numbered_list_item":
                text_content = self._extract_text_content(block.get("numbered_list_item", {}).get("rich_text", []))
                md_lines.append(f"1. {text_content}")

            elif block_type == "code":
                code_block = block.get("code", {})
                language = code_block.get("language", "")
                text_content = self._extract_text_content(code_block.get("rich_text", []))

                md_lines.append(f"```{language}")
                md_lines.append(text_content)
                md_lines.append("```")
                md_lines.append("")

            elif block_type == "to_do":
                todo_item = block.get("to_do", {})
                checked = todo_item.get("checked", False)
                text_content = self._extract_text_content(todo_item.get("rich_text", []))

                checkbox = "[x]" if checked else "[ ]"
                md_lines.append(f"- {checkbox} {text_content}")

            elif block_type == "quote":
                text_content = self._extract_text_content(block.get("quote", {}).get("rich_text", []))
                md_lines.append(f"> {text_content}")
                md_lines.append("")

        return "\n".join(md_lines)

    def _extract_text_content(self, rich_text_list):
        """
        Notionのリッチテキスト配列からプレーンテキストを抽出します。

        Args:
            rich_text_list: Notionのリッチテキスト配列

        Returns:
            抽出されたプレーンテキスト
        """
        if not rich_text_list:
            return ""

        return "".join([rt.get("text", {}).get("content", "") for rt in rich_text_list if "text" in rt])
