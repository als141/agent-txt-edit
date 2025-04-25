# -*- coding: utf-8 -*-
import os
import json
import asyncio
import re
import subprocess # MCPサーバー起動用に必要になる可能性
from pathlib import Path
from openai import AsyncOpenAI
# エラーハンドリングのために追加
from openai import BadRequestError
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.syntax import Syntax
from typing import List, Dict, Union, Optional, Tuple, Any, Literal, Callable, Awaitable # Callable, Awaitable を追加
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass, field
import difflib # 差分表示用にdifflibをインポート
import uuid # トレースID生成用 (オプション)

# --- Agents SDK ---
from agents import (
    Agent,
    Runner,
    RunConfig,
    function_tool,
    ModelSettings,
    RunContextWrapper,
    Tool,
    FunctionTool,
    # エラーハンドリング用
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
    UserError,
    # Handoff用
    handoff,
    # 動的プロンプト用
    # AgentInputInstructions, # v6で削除済み
)
# MCP用クラスを正しい場所からインポート
from agents.mcp.server import MCPServerStdio, MCPServer
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
# --------------------

console = Console()
# 環境変数の読み込み
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_TOKEN") # Notionトークンも読み込む
if not API_KEY:
    raise ValueError("OpenAI APIキーが設定されていません。.envファイルを確認してください。")
# Notionトークンがない場合は警告を出す
if not NOTION_API_KEY:
    console.print("[yellow]警告: Notion APIトークン (NOTION_TOKEN) が設定されていません。Notion連携機能は利用できません。[/yellow]")


# --- グローバル設定 ---
async_client = AsyncOpenAI(api_key=API_KEY)
MODEL = "gpt-4.1-mini" # ユーザーが指定したモデルに戻す

# --- Pydanticモデル定義 (Agentの出力型) ---
# (変更なし)
class SingleEdit(BaseModel):
    status: Literal["single_edit"] = Field(description="編集タイプ: 単一編集")
    old_context: str = Field(description="変更箇所を一意に特定するための、変更前の十分な文脈を含む文字列。ファイル内で正確に1回出現するはずです。")
    new_string: str = Field(description="変更後の新しい文字列")
    explanation: str = Field(description="なぜこの編集が必要かの説明")

class EditOperation(BaseModel):
    old_context: str = Field(description="変更箇所を一意に特定するための、変更前の十分な文脈を含む文字列。ファイル内で正確に1回出現し、他の編集箇所と重複しないはずです。")
    new_string: str = Field(description="変更後の新しい文字列")

class MultipleEdits(BaseModel):
    status: Literal["multiple_edits"] = Field(description="編集タイプ: 複数編集")
    edits: List[EditOperation] = Field(description="実行する編集操作のリスト")
    explanation: str = Field(description="なぜこれらの編集が必要かの説明")

class ReplaceAll(BaseModel):
    status: Literal["replace_all"] = Field(description="編集タイプ: 全体置換")
    new_content: str = Field(description="ファイル全体の新しい内容")
    explanation: str = Field(description="なぜ全体置換が必要かの説明")

class ClarificationNeeded(BaseModel):
    status: Literal["clarification_needed"] = Field(description="編集タイプ: 要確認")
    message: str = Field(description="ユーザーへの具体的な質問や確認事項")

class Conversation(BaseModel):
    status: Literal["conversation"] = Field(description="編集タイプ: 会話")
    message: str = Field(description="ユーザーへの応答メッセージ")

class Rejected(BaseModel):
    status: Literal["rejected"] = Field(description="編集タイプ: 拒否")
    message: str = Field(description="指示を拒否する理由")

AgentFinalOutput = Union[
    SingleEdit, MultipleEdits, ReplaceAll, ClarificationNeeded, Conversation, Rejected
]

# --- コンテキストクラス ---
@dataclass
class EditContext:
    """AgentやToolが共有するコンテキスト"""
    filepath: Path
    encoding: str
    current_content: Optional[str] = None
    last_applied_edit: Optional[AgentFinalOutput] = None
    last_proposal: Optional[AgentFinalOutput] = None

# --- ツール定義 ---
# grep_tool の実装関数 (変更なし)
async def _grep_tool_impl(
    ctx: RunContextWrapper[EditContext],
    search_pattern: str = Field(description="検索する文字列または正規表現パターン。"),
    file_content: str = Field(description="検索対象のファイル内容全体。"), # ★ file_content を引数で受け取る
    is_regex: Optional[bool] = Field(default=False, description="search_patternが正規表現かどうか。"),
    case_sensitive: Optional[bool] = Field(default=True, description="大文字と小文字を区別するかどうか。"),
    context_lines: Optional[int] = Field(default=0, description="マッチした行の前後何行を表示するか。"),
) -> Dict[str, Any]:
    """
    指定されたファイル内容(`file_content`)から、ユーザーが明示的に指定した検索パターン(`search_pattern`)を検索します。
    **重要:** このツールは、ユーザーが「〇〇を検索して」「△△があるか確認して」「grepを使って」のように、
    具体的な検索文字列やパターン、または検索行為自体を指示に含めている場合に**のみ**使用してください。
    単純な置換指示の際に、置換箇所を探すためだけには使用しないでください。
    """
    is_regex_actual = is_regex if is_regex is not None else False
    case_sensitive_actual = case_sensitive if case_sensitive is not None else True
    context_lines_actual = context_lines if context_lines is not None else 0
    console.print(f"[dim]ツール実行(実装): grep_tool (パターン: '{search_pattern}', 正規表現: {is_regex_actual}, 大小区別: {case_sensitive_actual}, 前後行: {context_lines_actual})[/dim]")
    if not file_content:
        file_content = ctx.context.current_content
        if not file_content: return {"success": False, "error": "検索対象のファイル内容が空です。"}
    lines = file_content.splitlines(); matches = []
    flags = 0 if case_sensitive_actual else re.IGNORECASE
    try:
        if is_regex_actual:
            compiled_pattern = re.compile(search_pattern, flags)
            for i, line in enumerate(lines):
                line_matches = list(compiled_pattern.finditer(line))
                if line_matches:
                    start_line = max(0, i - context_lines_actual); end_line = min(len(lines), i + context_lines_actual + 1)
                    context_text = lines[start_line:end_line]
                    matches.append({"line_number": i + 1, "line": line, "matches_in_line": [(m.start(), m.end(), m.group(0)) for m in line_matches], "context": context_text})
        else:
            search_term = search_pattern if case_sensitive_actual else search_pattern.lower()
            for i, line in enumerate(lines):
                compare_line = line if case_sensitive_actual else line.lower(); start_index = 0; line_matches_indices = []
                while True:
                    found_index = compare_line.find(search_term, start_index)
                    if found_index == -1: break
                    end_index = found_index + len(search_pattern)
                    line_matches_indices.append((found_index, end_index, line[found_index:end_index])); start_index = end_index
                if line_matches_indices:
                    start_line = max(0, i - context_lines_actual); end_line = min(len(lines), i + context_lines_actual + 1)
                    context_text = lines[start_line:end_line]
                    matches.append({"line_number": i + 1, "line": line, "matches_in_line": line_matches_indices, "context": context_text})
        if matches: return {"success": True, "matches": matches, "count": len(matches)}
        else: return {"success": True, "matches": [], "count": 0, "message": "パターンにマッチする箇所は見つかりませんでした。"}
    except re.error as e: error_msg = f"正規表現エラー: {e}"; console.print(f"[bold red]エラー (grep_tool): {error_msg}[/]"); return {"success": False, "error": error_msg}
    except Exception as e: error_msg = f"検索中に予期せぬエラー: {e}"; console.print(f"[bold red]エラー (grep_tool): {error_msg}[/]"); return {"success": False, "error": error_msg}
grep_tool = function_tool(_grep_tool_impl)

# --- 動的プロンプト生成関数 ---
def create_dynamic_instructions(base_prompt: str) -> Callable[[RunContextWrapper[EditContext], Agent[EditContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[EditContext], agent: Agent[EditContext]) -> str:
        content_to_agent = ctx.context.current_content or ""
        MAX_CONTENT_LENGTH = 10000
        if len(content_to_agent) > MAX_CONTENT_LENGTH: content_to_agent = content_to_agent[:MAX_CONTENT_LENGTH] + "\n... (以下省略)"
        previous_action_info = ""
        if ctx.context.last_applied_edit:
             try: applied_edit_json = ctx.context.last_applied_edit.model_dump_json(indent=2); previous_action_info = f"""【直前に適用された編集】:\n```json\n{applied_edit_json}\n```"""
             except Exception: previous_action_info = "【直前に適用された編集】: (情報取得失敗)"
        last_proposal_info = ""
        if ctx.context.last_proposal:
             try: proposal_json = ctx.context.last_proposal.model_dump_json(indent=2); last_proposal_info = f"""【前回のAI提案】:\n```json\n{proposal_json}\n```"""
             except Exception: last_proposal_info = "【前回のAI提案】: (情報取得失敗)"
        full_prompt = f"""{base_prompt}

--- 追加コンテキスト情報 ---
【現在のファイル内容】(ファイルパス: {ctx.context.filepath}):
```
{content_to_agent}
```
{previous_action_info}
{last_proposal_info}
--- 追加コンテキスト情報終 ---

**あなたの応答:**
上記の指示と追加コンテキスト情報、そして入力として与えられる会話履歴全体を考慮し、タスクを実行してください。
応答は指示されたJSON形式で出力するか、必要であれば他のエージェントへのハンドオフツールを呼び出してください。
"""
        return full_prompt
    return dynamic_instructions_func

# --- エージェント定義 (動的プロンプトを使用) ---

# ★ 相互ハンドオフのために、先に GeneralEditAgent を仮定義
general_edit_agent: Optional[Agent[EditContext]] = None

# 1. 置換担当エージェント (ReplaceAgent) - ★ 相互ハンドオフ対応
REPLACE_AGENT_BASE_PROMPT = """
あなたはテキスト内の特定の文字列を置換する専門のエージェントです。
ユーザー指示（会話履歴の最後のメッセージ）と追加コンテキスト情報（現在のファイル内容など）が与えられます。
あなたは `GeneralEditAgent` に処理をハンドオフできます。

**あなたのタスク:**
1.  会話履歴の最後のユーザー指示を分析し、**明確な置換元 (`old_string`) と置換先 (`new_string`)** が指定されているか確認します。
2.  **もし指示が明確な置換ではない、または曖昧さを含む場合 (例: 「Aをいい感じにBにして」など):**
    * **このタスクはあなたの専門外です。**
    * `transfer_to_GeneralEditAgent` ツールを呼び出して、`GeneralEditAgent` に処理を依頼してください。ハンドオフする理由（例：「指示が曖昧なため、一般編集エージェントに依頼します」）を明確に述べてください。
3.  **指示が明確な置換の場合:**
    * **追加コンテキスト情報の【現在のファイル内容】を直接分析**し、指定された `old_string` がファイル内に存在するか、何箇所存在するかを判断します。
    * **`grep_tool` の使用は限定的です:** ユーザーが明示的に検索を指示した場合や、多数の候補から判断に迷う場合にのみ使用してください。単純な置換箇所探索には使わないでください。
    * 分析結果に基づき、最適な編集提案 (`single_edit` または `multiple_edits`) を作成します。
    * もし置換元が見つからない場合は `clarification_needed` を返します。
4.  `old_context` 生成ルールは厳守してください。
5.  応答は必ず `AgentFinalOutput` 型のJSON形式、または `transfer_to_GeneralEditAgent` のツール呼び出しのいずれかです。
"""
replace_agent = Agent[EditContext](
    name="ReplaceAgent",
    handoff_description="ユーザーが明示的に指定した特定の文字列の置換を専門に行うエージェント。曖昧な指示はGeneralEditAgentに渡します。",
    instructions=create_dynamic_instructions(REPLACE_AGENT_BASE_PROMPT),
    model=MODEL,
    tools=[grep_tool],
    # ★ GeneralEditAgent へのハンドオフを追加 (遅延評価)
    handoffs=lambda: [general_edit_agent] if general_edit_agent else [],
    output_type=AgentFinalOutput,
)

# 2. 一般編集担当エージェント (GeneralEditAgent) - ★ 相互ハンドオフ対応
GENERAL_EDIT_AGENT_BASE_PROMPT = """
あなたはテキストの一般的な編集（誤字脱字修正、表現改善、内容追加・削除、スタイル変更、要約、構造変更など）や、会話履歴に基づいた操作（元に戻すなど）を行う専門のエージェントです。
ユーザー指示（会話履歴の最後のメッセージ）と追加コンテキスト情報（現在のファイル内容、直前の編集、前回の提案）が与えられます。
あなたは `ReplaceAgent` に処理をハンドオフできます。

**あなたのタスク:**
1.  会話履歴全体、特に最後のユーザー指示と、それに対する直前のAIの応答（【前回のAI提案】や【直前に適用された編集】）を注意深く分析し、ユーザーの意図を深く理解します。
2.  **もし指示が非常に明確な「AをBに置換」であり、曖昧さが全くない場合:**
    * このタスクは `ReplaceAgent` の方が適している可能性があります。
    * `transfer_to_ReplaceAgent` ツールを呼び出して、`ReplaceAgent` に処理を依頼することを検討してください。ハンドオフする理由（例：「明確な置換指示のため、置換専門エージェントに依頼します」）を明確に述べてください。**ただし、少しでも曖昧さがある場合や、置換以外の要素が含まれる場合は、自分で処理してください。**
3.  **「元に戻す」系の指示への対応:**
    * 【直前に適用された編集】が存在する場合、その編集操作を逆転させる提案 (`SingleEdit` または `MultipleEdits`) を作成します。
    * 直前が `ReplaceAll` の場合は元に戻せないため `clarification_needed` を返します。
    * 【直前に適用された編集】がない場合は `clarification_needed` を返します。
4.  **その他の一般編集指示への対応:**
    * 【現在のファイル内容】と会話履歴を考慮し、編集の範囲と性質に基づいて、最適な編集提案 (`single_edit`, `multiple_edits`, `replace_all`) を作成します。
    * 指示が編集要求でない場合は `conversation` を返します。
    * 指示が曖昧すぎる場合は `clarification_needed` を返します。
    * 実行不可能な場合は `rejected` を返します。
5.  `old_context` 生成ルールは厳守してください。
6.  応答は必ず `AgentFinalOutput` 型のJSON形式、または `transfer_to_ReplaceAgent` のツール呼び出しのいずれかです。
7.  **注意:** あなたは `grep_tool` を使用できません。
"""
# ★ GeneralEditAgent を正式に定義
general_edit_agent = Agent[EditContext](
    name="GeneralEditAgent",
    handoff_description="表現修正、内容追加・削除、スタイル変更、要約、構造変更、**特に『元に戻す』操作**を含む、一般的な編集タスクを専門に行うエージェント。明確な置換はReplaceAgentに渡すことがあります。",
    instructions=create_dynamic_instructions(GENERAL_EDIT_AGENT_BASE_PROMPT),
    model=MODEL,
    tools=[],
    # ★ ReplaceAgent へのハンドオフを追加
    handoffs=[replace_agent],
    output_type=AgentFinalOutput,
)

# 3. 司令塔エージェント (TriageAgent) - プロンプト微調整
TRIAGE_AGENT_BASE_PROMPT = """
あなたはテキスト編集や関連タスクのリクエストを受け付け、内容を分析して最適な処理（専門エージェントへのハンドオフ、MCPツールの利用、自身での応答）を判断する司令塔エージェントです。
ユーザー指示（会話履歴の最後のメッセージ）と追加コンテキスト情報（現在のファイル内容、直前の編集、前回の提案）が与えられます。
あなたは以下の外部ツールやハンドオフ先を利用できます:
- ファイルシステム操作用MCPサーバー (例: `filesystem.*`)
- Notion連携用MCPサーバー (Light版) (例: `notionLight.uploadMarkdown`)
- Notion連携用MCPサーバー (公式API版) (例: `notionApi.*`)
- `ReplaceAgent`: **ユーザーが明確に「AをBに置換して」と具体的な文字列を指定した場合**の処理を担当。
- `GeneralEditAgent`: 上記以外の**一般的な編集指示**や、**特に「元に戻して」「取り消して」のような会話の文脈に基づく指示**を担当。
**注意:** `ReplaceAgent` と `GeneralEditAgent` は、必要に応じて互いにハンドオフすることがあります。

**あなたのタスク:**
1.  **会話履歴全体**と追加コンテキスト情報を分析し、**最新のユーザー指示**がどのタイプに該当するか**慎重に判断**します。
    * **Notion操作 (書き込み/アップロード)**
    * **Notion操作 (読み取り/検索/その他)**
    * **ファイル操作**
    * **明確な文字列置換:** 例：「リンゴをバナナに置換」
    * **一般編集/文脈依存:** 表現修正、内容追加・削除、スタイル変更、要約、構造変更、誤字脱字修正など。**特に、「元に戻して」「さっきのを修正」「アンドゥ」のような、直前の操作や会話の流れに依存する指示はこのカテゴリです。**
    * **会話/質問:** 上記いずれでもない場合。
    * **不明/拒否:** 指示が不明瞭すぎる、実行不可能、または不適切な場合。
2.  判断結果に基づいて、以下のいずれかのアクションを実行します。
    * **Notion操作 (書き込み/アップロード)の場合:** **NotionMCP Light** の `uploadMarkdown` ツールを呼び出します。`filepath`引数には【現在のファイル内容】のパス ({filepath}) を渡します。
    * **Notion操作 (読み取り/検索/その他)の場合:** **公式Notion API** の適切なツールを呼び出します。
    * **ファイル操作の場合:** 適切な ファイルシステム MCP ツール を呼び出します。
    * **明確な文字列置換の場合:** `ReplaceAgent` にハンドオフします。ユーザーが検索を指示した場合のみ、その旨を伝えてください。
    * **一般編集/文脈依存の場合:** `GeneralEditAgent` にハンドオフします。特に「元に戻す」系の指示の場合は、【直前に適用された編集】の情報が重要であることを明確に伝えてください（ただし、情報はプロンプト経由で渡されます）。
    * **会話/質問の場合:** あなた自身で応答メッセージを生成し、`conversation` 形式で出力します。
    * **不明/拒否の場合:** `clarification_needed` または `rejected` 形式で出力します。
3.  ハンドオフする場合、ハンドオフ先のツール (`transfer_to_ReplaceAgent` など) を呼び出します。SDKが会話履歴を渡すので、あなたは追加の指示があればそれを渡すだけで構いません。
4.  応答は、自身で応答する場合 (`conversation`, `clarification_needed`, `rejected`) は `AgentFinalOutput` 型のJSON形式で、ハンドオフやMCPツール利用の場合は対応するツール呼び出しとなります。
"""
# TriageAgentの定義は main 関数内で動的プロンプトと共に行う

# --- ヘルパー関数 ---
# (変更なし)
def get_file_path() -> Optional[Path]:
    while True:
        try:
            filepath_str = Prompt.ask("[bold cyan]編集したいファイルのパスを入力してください[/]", default="")
            if not filepath_str:
                if Prompt.ask("[yellow]パスが入力されていません。終了しますか？ (y/n)[/]", choices=["y", "n"], default="y") == "y": return None
                else: continue
            filepath = Path(filepath_str).resolve()
            if filepath.is_file(): return filepath
            else: console.print(f"[bold red]エラー: ファイルが見つかりません: {filepath}[/]")
        except Exception as e: console.print(f"[bold red]エラー: 無効なパスです: {e}[/]")

def determine_encoding(filepath: Path) -> str:
    try:
        with open(filepath, 'rb') as f:
            bom = f.read(3)
            if bom == b'\xef\xbb\xbf': console.print("[dim]UTF-8 (BOM付き) 検出。[/dim]"); return 'utf-8-sig'
        filepath.read_text(encoding='utf-8'); console.print("[dim]UTF-8 (BOMなし) 検出。[/dim]"); return 'utf-8'
    except UnicodeDecodeError:
        try: filepath.read_text(encoding='shift_jis'); console.print("[dim]Shift-JIS 検出。[/dim]"); return 'shift_jis'
        except Exception: console.print("[yellow]警告: UTF-8/Shift-JIS判別失敗。デフォルトUTF-8。[/yellow]"); return 'utf-8'
    except Exception as e: console.print(f"[yellow]警告: エンコーディング判別エラー({e})。デフォルトUTF-8。[/yellow]"); return 'utf-8'

def read_file_content(filepath: Path, encoding: str) -> Optional[str]:
    console.print(f"[dim]ファイル読込中... ({filepath}, {encoding})[/dim]")
    try: return filepath.read_text(encoding=encoding)
    except Exception as e: console.print(f"[bold red]エラー: ファイル読込失敗 ({filepath}, {encoding}): {e}[/]"); return None

def ask_confirmation(prompt_message: str) -> str:
    while True:
        response = Prompt.ask(f"{prompt_message} ([bold green]y[/]/[bold red]n[/]/[bold yellow]フィードバック[/])", default="").strip()
        response_lower = response.lower()
        if response_lower == 'y': return 'y'
        elif response_lower == 'n': return 'n'
        elif response: return response
        else: console.print("[yellow]入力なし。'y','n',フィードバックを入力。[/yellow]")

def ask_multiple_edit_confirmation(num_edits: int) -> Union[str, List[int]]:
    while True:
        user_choice_str = Prompt.ask(f"\n適用編集番号(1-{num_edits})をカンマ区切り(例:1,3)。全適用:'all',非適用:'n',FBはそのまま:", default="").strip()
        if not user_choice_str: console.print("[yellow]入力なし。番号/all/n/FBを入力。[/yellow]"); continue
        if user_choice_str.lower() == 'n': return 'n'
        elif user_choice_str.lower() == 'all': return 'all'
        try:
            selected_ids = set(); parts = user_choice_str.replace(' ', '').split(','); is_numeric_input = True
            for part in parts:
                if not part.isdigit(): is_numeric_input = False; break
                num = int(part)
                if not (1 <= num <= num_edits): console.print(f"[red]エラー:番号{num}無効(1-{num_edits})。[/red]"); is_numeric_input = False; break
                selected_ids.add(num)
            if is_numeric_input:
                if not selected_ids:
                     if Prompt.ask("[yellow]有効番号なし。キャンセル？(y/n)[/]", choices=["y", "n"], default="y") == "y": return 'n'
                     else: continue
                return sorted(list(selected_ids))
            else: return user_choice_str
        except ValueError: return user_choice_str

def perform_write(filepath: Path, content: str, encoding: str) -> bool:
    console.print(f"[dim]ファイル書込実行中... ({filepath}, {encoding})[/dim]")
    try:
        original_stat = None
        try: original_stat = filepath.stat()
        except OSError: pass
        filepath.write_text(content, encoding=encoding)
        if original_stat and hasattr(os, 'chmod'):
            try: os.chmod(filepath, original_stat.st_mode)
            except OSError as e: console.print(f"[yellow]警告:パーミッション復元失敗:{e}[/yellow]")
        console.print("[green]ファイル書込成功。[/green]"); return True
    except Exception as e: console.print(f"[bold red]エラー:ファイル書込失敗:{e}[/]"); return False

def display_diff(old_text: str, new_text: str):
    old_lines = old_text.splitlines(keepends=True); new_lines = new_text.splitlines(keepends=True)
    line_diff_ratio = abs(len(old_lines) - len(new_lines)) / max(len(old_lines), len(new_lines), 1)
    max_total_lines = max(len(old_lines), len(new_lines))
    if line_diff_ratio > 0.8 or max_total_lines > 300:
        console.print("[bold blue]--- 新内容(全体表示) ---[/]")
        display_content = new_text[:5000] + ("\n...(省略)" if len(new_text) > 5000 else "")
        syntax = Syntax(display_content, "text", theme="default", line_numbers=False); console.print(syntax)
        console.print("[bold blue]--- 新内容終 ---[/]"); return
    diff = difflib.unified_diff(old_lines, new_lines, fromfile='変更前', tofile='変更後', lineterm='')
    console.print("[bold]--- 差分表示 ---[/]"); output_lines = 0; max_lines_to_show = 100; has_diff = False
    for line in diff:
        has_diff = True
        if output_lines >= max_lines_to_show: console.print("[dim]...(差分省略)[/dim]"); break
        if line.startswith('+') and not line.startswith('+++'): console.print(f"[green]{line.rstrip()}[/]")
        elif line.startswith('-') and not line.startswith('---'): console.print(f"[red]{line.rstrip()}[/]")
        elif line.startswith('@@'): console.print(f"[cyan]{line.rstrip()}[/]")
        else: console.print(line.rstrip())
        output_lines += 1
    if not has_diff: console.print("[yellow]内容変更なし。[/yellow]")
    console.print("[bold]--- 差分表示終 ---[/]")

# --- メイン実行部分 ---
async def main():
    console.print("[bold magenta]AIテキスト編集エージェントへようこそ！ (MCP+Notion連携・改良版 v7)[/]") # バージョン表示変更
    filepath = get_file_path()
    if not filepath: console.print("[bold magenta]終了します。[/]"); return

    file_encoding = determine_encoding(filepath)
    console.print(f"[green]編集対象ファイル: {filepath} (エンコーディング: {file_encoding})[/]")

    initial_content = read_file_content(filepath, file_encoding)
    if initial_content is None: console.print("[bold red]初期ファイル読み込み失敗。終了します。[/]"); return

    edit_context = EditContext(
        filepath=filepath, encoding=file_encoding, current_content=initial_content,
        last_applied_edit=None, last_proposal=None
    )

    # --- MCPサーバーの設定 ---
    # (変更なし、パスは要確認)
    fs_mcp_command = "npx"; fs_mcp_args = ["-y", "@modelcontextprotocol/server-filesystem", "."]; fs_mcp_server: Optional[MCPServer] = None
    notion_mcp_light_dir = "../notion-mcp-light"; notion_light_mcp_command = "uv"; notion_light_mcp_args = ["run", "--directory", notion_mcp_light_dir, "python", "-m", "src.main"]; notion_light_mcp_env = {"NOTION_TOKEN": NOTION_API_KEY} if NOTION_API_KEY else {}; notion_light_mcp_server: Optional[MCPServer] = None
    official_notion_mcp_command = "npx"; official_notion_mcp_args = ["-y", "@notionhq/notion-mcp-server"]; official_notion_headers = json.dumps({"Authorization": f"Bearer {NOTION_API_KEY}", "Notion-Version": "2022-06-28"}) if NOTION_API_KEY else "{}"; official_notion_mcp_env = {"OPENAPI_MCP_HEADERS": official_notion_headers}; official_notion_mcp_server: Optional[MCPServer] = None

    try:
        fs_mcp_server_instance = MCPServerStdio(params={"command": fs_mcp_command, "args": fs_mcp_args})
        async with fs_mcp_server_instance as fs_server:
            fs_mcp_server = fs_server; console.print("[cyan]ファイルシステムMCPサーバー接続完了。[/cyan]")
            if NOTION_API_KEY:
                try:
                    notion_light_mcp_instance = MCPServerStdio(params={"command": notion_light_mcp_command, "args": notion_light_mcp_args, "env": notion_light_mcp_env})
                    async with notion_light_mcp_instance as notion_light_server:
                        notion_light_mcp_server = notion_light_server; console.print("[cyan]NotionMCP Lightサーバー接続完了。[/cyan]")
                        try:
                            official_notion_mcp_instance = MCPServerStdio(params={"command": official_notion_mcp_command, "args": official_notion_mcp_args, "env": official_notion_mcp_env})
                            async with official_notion_mcp_instance as official_notion_server:
                                official_notion_mcp_server = official_notion_server; console.print("[cyan]公式Notion MCPサーバー接続完了。[/cyan]")
                                await run_main_loop(edit_context, fs_mcp_server, notion_light_mcp_server, official_notion_mcp_server)
                        except Exception as e:
                            console.print(f"[bold red]公式Notion MCP接続失敗: {e}[/]"); console.print("[yellow]公式Notion連携無効。[/yellow]")
                            await run_main_loop(edit_context, fs_mcp_server, notion_light_mcp_server, None)
                except FileNotFoundError:
                     console.print(f"[bold red]エラー: NotionMCP Light コマンド '{notion_light_mcp_command}' 未検出。[/]"); console.print(f"[dim]試行Dir: {notion_mcp_light_dir}[/dim]"); console.print("[yellow]Notion連携無効。[/yellow]")
                     await run_main_loop(edit_context, fs_mcp_server, None, None)
                except Exception as e:
                    console.print(f"[bold red]NotionMCP Light接続失敗: {e}[/]"); console.print(f"[dim]試行Dir: {notion_mcp_light_dir}[/dim]"); console.print("[yellow]Notion連携無効。[/yellow]")
                    await run_main_loop(edit_context, fs_mcp_server, None, None)
            else: await run_main_loop(edit_context, fs_mcp_server, None, None)
    except FileNotFoundError:
        console.print(f"[bold red]エラー: FS MCP コマンド '{fs_mcp_command}' 未検出。[/]")
        if Prompt.ask("[yellow]MCP機能なしで続行？ (y/n)[/]", choices=["y", "n"], default="n") == "y":
             console.print("[yellow]MCP機能なしで実行。[/yellow]"); await run_main_loop(edit_context, None, None, None)
        else: console.print("[red]MCPサーバーエラーのため終了。[/red]")
    except Exception as e:
        console.print(f"[bold red]FS MCPサーバーエラー: {e}[/]")
        if Prompt.ask("[yellow]MCP機能なしで続行？ (y/n)[/]", choices=["y", "n"], default="n") == "y":
            console.print("[yellow]MCP機能なしで実行。[/yellow]"); await run_main_loop(edit_context, None, None, None)
        else: console.print("[red]MCPサーバーエラーのため終了。[/red]")
    finally: console.print("[cyan]プログラム終了。[/cyan]")


# --- メインループ部分を関数化 ---
async def run_main_loop(
    edit_context: EditContext,
    fs_mcp_server: Optional[MCPServer],
    notion_light_mcp_server: Optional[MCPServer],
    official_notion_mcp_server: Optional[MCPServer]
):
    """エージェントとの対話ループを実行する関数"""

    active_mcp_servers = []
    if fs_mcp_server: active_mcp_servers.append(fs_mcp_server)
    if notion_light_mcp_server: active_mcp_servers.append(notion_light_mcp_server)
    if official_notion_mcp_server: active_mcp_servers.append(official_notion_mcp_server)

    # ★ TriageAgent の instructions も動的に生成
    triage_agent_final = Agent[EditContext](
        name="TriageAgent",
        instructions=create_dynamic_instructions(TRIAGE_AGENT_BASE_PROMPT), # ★ 動的プロンプト
        model=MODEL,
        tools=[],
        # ★ TriageAgentは専門エージェントにハンドオフするだけ
        handoffs=[replace_agent, general_edit_agent],
        mcp_servers=active_mcp_servers,
        output_type=AgentFinalOutput, # 直接応答する場合の型
    )

    run_config = RunConfig(
        workflow_name="AdvancedFileEditWorkflow-v7", # バージョン更新
        max_turns=15 # ★ 最大ターン数を設定してループ制御
    )

    total_tokens_used = 0
    filepath = edit_context.filepath
    file_encoding = edit_context.encoding

    # ★ 会話履歴リスト
    conversation_history: List[Dict[str, Any]] = []

    while True:
        console.rule()
        try:
            current_user_input = Prompt.ask(f"[bold yellow]{filepath.name}>[/]", default="")

            if not current_user_input.strip(): continue
            if current_user_input.lower() in ["/quit", "/exit", "終了", "exit", "quit"]: break
            if current_user_input.lower() == "/show":
                current_content_display = edit_context.current_content
                if current_content_display is not None:
                    console.print("\n[bold cyan]--- 現在のファイル内容 (メモリ上) ---[/]")
                    try: syntax = Syntax(current_content_display, filepath.suffix.lstrip('.'), theme="default", line_numbers=True); console.print(syntax)
                    except Exception: syntax = Syntax(current_content_display, "text", theme="default", line_numbers=True); console.print(syntax)
                    console.print("[bold cyan]--- 現在のファイル内容終 ---[/]")
                else: console.print("[yellow]ファイル内容がメモリ上にありません。[/yellow]")
                continue
            if current_user_input.lower() == "/reload":
                console.print("[dim]ファイルを再読み込みします...[/dim]")
                reloaded_content = read_file_content(filepath, edit_context.encoding)
                if reloaded_content is not None:
                    edit_context.current_content = reloaded_content
                    console.print("[green]ファイルの再読み込みが完了しました。[/green]")
                    edit_context.last_applied_edit = None; edit_context.last_proposal = None
                    conversation_history = [] # ★ リロードしたら会話履歴もリセット
                else: console.print("[red]ファイルの再読み込みに失敗しました。[/red]")
                continue

            console.print("[dim]AIエージェント(司令塔)が思考中...[/dim]")

            # --- Agentへの入力メッセージ準備 (会話履歴形式) ---
            # 1. 今回のユーザー入力を履歴に追加
            conversation_history.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": current_user_input}]
            })

            # 2. ★ EditContext の last_proposal を更新 (動的プロンプト生成用)
            #    ループの最後で設定された前回の提案を使う
            #    (このタイミングで設定するのが適切か要検討。Agent実行前に設定すべき)
            #    edit_context.last_proposal = last_agent_proposal_obj # これは間違い

            # --- Agent実行 ---
            # ★ input には会話履歴リスト全体を渡す
            result = await Runner.run(
                starting_agent=triage_agent_final,
                input=conversation_history, # ★ 会話履歴リストを渡す
                context=edit_context,      # コンテキスト（ファイル内容等アクセス用）
                run_config=run_config,     # max_turns などを含む
                # max_turns は run_config で設定するのでここでは不要
            )

            # --- トークン数表示 ---
            run_tokens = 0
            if result.raw_responses:
                for resp in result.raw_responses:
                    usage = getattr(resp, 'usage', None)
                    if usage and hasattr(usage, 'total_tokens'): run_tokens += usage.total_tokens
            if run_tokens > 0:
                total_tokens_used += run_tokens
                console.print(f"[cyan]今回トークン(推定): {run_tokens}, 累計: {total_tokens_used}[/]")

            # --- Agentの出力を取得・検証 ---
            agent_output: Optional[AgentFinalOutput] = None
            mcp_tool_output: Optional[str] = None
            assistant_response_content: List[Dict[str, Any]] = [] # 履歴用

            try:
                final_output_raw = result.final_output
                if isinstance(final_output_raw, AgentFinalOutput.__args__): agent_output = final_output_raw
                elif isinstance(final_output_raw, str):
                    try: parsed = json.loads(final_output_raw); agent_output = AgentFinalOutput(**parsed) # type: ignore
                    except (json.JSONDecodeError, ValidationError): agent_output = Conversation(status="conversation", message=final_output_raw)
                elif isinstance(final_output_raw, list) and final_output_raw:
                     first_output_item = final_output_raw[0]
                     if first_output_item.get("type") == "message" and first_output_item.get("role") == "assistant":
                          content_list = first_output_item.get("content", [])
                          assistant_response_content = content_list
                          if content_list and content_list[0].get("type") == "output_text":
                               text_output = content_list[0].get("text", "")
                               try: parsed = json.loads(text_output); agent_output = AgentFinalOutput(**parsed) # type: ignore
                               except (json.JSONDecodeError, ValidationError): agent_output = Conversation(status="conversation", message=text_output)
                          else: agent_output = Conversation(status="conversation", message="(テキスト応答なし)")
                     else: raise TypeError(f"最終出力が予期しないリスト形式: {final_output_raw}")
                elif final_output_raw is None and result.new_items:
                     last_item = result.new_items[-1]
                     if last_item.type == "tool_call_output_item":
                         mcp_tool_output = str(getattr(last_item, 'output', 'MCPツール出力取得失敗'))
                         console.print(f"[blue]MCPツール実行結果:\n{mcp_tool_output[:500]}{'...' if len(mcp_tool_output)>500 else ''}[/blue]")
                         tool_call_id = getattr(last_item, 'tool_call_id', None)
                         if tool_call_id: conversation_history.append({"type": "message", "role": "tool", "content": [{"type": "tool_result", "tool_call_id": tool_call_id, "result": mcp_tool_output}]})
                         edit_context.last_applied_edit = None; edit_context.last_proposal = None
                         continue
                     elif last_item.type == "handoff_output_item":
                         console.print(f"[cyan]ハンドオフ発生: {getattr(last_item.source_agent,'name','?')} -> {getattr(last_item.target_agent,'name','?')}[/cyan]")
                         # ハンドオフ発生時は通常、次のエージェントが応答するので、ここでは履歴に追加せずループを続ける
                         # （もし最終応答がない場合は下のNoneチェックで捕捉される）
                         # edit_context.last_applied_edit = None; edit_context.last_proposal = None # ハンドオフ時はリセットしない方が良い場合も？
                         continue # ★ ハンドオフ後は次のAgentの処理を待つ
                elif final_output_raw is None:
                     console.print("[yellow]AIからの応答なし。[/yellow]")
                     edit_context.last_applied_edit = None; edit_context.last_proposal = None
                     continue
                else: raise TypeError(f"最終出力が予期しない型: {type(final_output_raw)}")

                edit_context.last_proposal = agent_output # ★ 提案内容をコンテキストに保存

                if not assistant_response_content: # final_outputがリスト形式でなかった場合など
                    if isinstance(agent_output, (SingleEdit, MultipleEdits, ReplaceAll, ClarificationNeeded, Conversation, Rejected)):
                         try: assistant_response_content.append({"type": "output_text", "text": agent_output.model_dump_json(indent=2)})
                         except Exception: assistant_response_content.append({"type": "output_text", "text": str(agent_output)})
                    else: assistant_response_content.append({"type": "output_text", "text": str(agent_output)})

                conversation_history.append({
                    "type": "message", "role": "assistant", "content": assistant_response_content
                }) # ★ AI応答を会話履歴に追加

            except (ValidationError, TypeError, AttributeError, json.JSONDecodeError) as e:
                console.print(f"[bold red]エラー: Agent応答のパース/処理失敗。[/]")
                console.print(f"エラー詳細: {e}")
                console.print(f"受信データ: {result.final_output if 'result' in locals() else 'N/A'}")
                edit_context.last_proposal = None # エラー時は提案リセット
                continue

            # --- Agentの出力に応じた処理 ---
            current_content = edit_context.current_content
            if current_content is None and agent_output and agent_output.status not in ["conversation", "clarification_needed", "rejected"]:
                console.print("[bold red]エラー: 編集対象ファイル内容読み込み不可。[/bold red]")
                edit_context.last_proposal = None; continue

            if mcp_tool_output: continue # MCPツール実行後は次のループへ
            if agent_output is None:
                 console.print("[yellow]AIからの有効な提案なし。[/yellow]")
                 edit_context.last_proposal = None; continue

            # --- 編集提案の処理 ---
            edit_applied = False # このターンで編集が適用されたか

            # 1. 単一編集提案
            if agent_output.status == "single_edit":
                console.print(f"\n[bold green]AI ({result.last_agent.name}) 単一編集提案:[/]")
                console.print(f"説明: {agent_output.explanation}")
                console.print(f"[red]- 前:[/]\n'{agent_output.old_context}'"); console.print(f"[blue]+ 後:[/]\n'{agent_output.new_string}'")
                if current_content is None: console.print("[bold red]エラー: ファイル内容なし。[/bold red]"); continue
                try: escaped_context = re.escape(agent_output.old_context); matches = list(re.finditer(escaped_context, current_content)); count = len(matches)
                except Exception as e: console.print(f"[bold red]変更元検索エラー: {e}[/]"); continue
                if count == 0: console.print(f"[bold red]エラー: 変更元テキスト未検出。[/]"); console.print("[yellow]'/reload'か修正依頼を。[/yellow]"); continue
                elif count > 1: console.print(f"[bold yellow]警告: 変更元が複数 ({count}箇所)。[/]"); console.print("[yellow]AIに一意な箇所を指定させてください。[/yellow]"); continue

                user_choice = ask_confirmation("\nこの編集を適用しますか？")
                if user_choice == 'y':
                    try:
                        new_content = current_content.replace(agent_output.old_context, agent_output.new_string, 1)
                        if perform_write(filepath, new_content, file_encoding):
                            edit_context.current_content = new_content; edit_context.last_applied_edit = agent_output
                            edit_context.last_proposal = None; edit_applied = True
                            conversation_history.append({"role": "user", "content": [{"type": "input_text", "text": "y (承認)"}]})
                        else: console.print("[red]書込失敗。[/red]")
                    except Exception as e: console.print(f"[bold red]置換/書込エラー: {e}[/]")
                elif user_choice == 'n':
                    console.print("[yellow]編集キャンセル。[/yellow]")
                    conversation_history.append({"role": "user", "content": [{"type": "input_text", "text": "n (拒否)"}]})
                else: # フィードバック
                    console.print("[yellow]フィードバック受付。[/yellow]")
                    # フィードバックは次のループで user メッセージとして conversation_history に追加される

            # 2. 複数編集提案
            elif agent_output.status == "multiple_edits":
                console.print(f"\n[bold green]AI ({result.last_agent.name}) 複数編集提案:[/]")
                console.print(f"説明: {agent_output.explanation}")
                if current_content is None: console.print("[bold red]エラー: ファイル内容なし。[/bold red]"); continue
                validated_edits_with_indices, problems = [], []
                processed_indices, temp_content = set(), current_content
                for i, edit in enumerate(agent_output.edits):
                    old, new = edit.old_context, edit.new_string; edit_label = f"編集 {i+1} ({old[:20].strip()}...)"
                    try:
                        escaped_context = re.escape(old); indices = [(m.start(), m.end()) for m in re.finditer(escaped_context, temp_content)]
                        if len(indices) == 0: original_indices = [(m.start(), m.end()) for m in re.finditer(escaped_context, current_content)]; problem_msg = f"- {edit_label}: 変更元未検出" if not original_indices else f"- {edit_label}: 変更元あり(競合可能性)"; problems.append(f"{problem_msg}:\n  ---\n  '{old}'\n  ---"); continue
                        elif len(indices) > 1: problems.append(f"- {edit_label}: 変更元複数 ({len(indices)}箇所):\n  ---\n  '{old}'\n  ---"); continue
                        else:
                             start_index, end_index = indices[0]; current_range = range(start_index, end_index)
                             is_overlapping = any(not set(current_range).isdisjoint(set(pr)) for pr in processed_indices)
                             if is_overlapping: problems.append(f"- {edit_label}: 他編集と重複:\n  ---\n  '{old}'\n  ---"); continue
                             else: validated_edits_with_indices.append({"start": start_index, "end": end_index, "old": old, "new": new, "original_index": i + 1}); processed_indices.add(frozenset(current_range)); temp_content = temp_content[:start_index] + new + temp_content[end_index:]
                    except Exception as e: problems.append(f"- {edit_label}: 検証エラー: {e}"); continue
                if problems: console.print("[bold red]エラー: 提案編集に問題あり:[/]"); [console.print(p) for p in problems]; console.print("[yellow]適用可能編集のみ表示。問題は修正依頼を。[/yellow]")
                if not validated_edits_with_indices: console.print("[yellow]適用可能編集案なし。[/yellow]"); continue
                validated_edits_with_indices.sort(key=lambda item: item["original_index"])
                console.print("\n[bold green]適用可能な編集案:[/]")
                for i, edit_info in enumerate(validated_edits_with_indices):
                    edit_info["display_id"] = i + 1
                    console.print(f"\n--- 編集 [bold]{edit_info['display_id']}[/bold] (元提案 {edit_info['original_index']}) ---")
                    console.print(f"[red]- 前:[/]\n'{edit_info['old']}'"); console.print(f"[blue]+ 後:[/]\n'{edit_info['new']}'")

                user_choice = ask_multiple_edit_confirmation(len(validated_edits_with_indices))
                selected_edits_to_apply, is_feedback, feedback_text = [], False, ""
                if isinstance(user_choice, list):
                    selected_ids = set(user_choice); selected_edits_to_apply = [e for e in validated_edits_with_indices if e["display_id"] in selected_ids]
                    if selected_edits_to_apply: console.print(f"[green]編集 {', '.join(map(str, sorted(user_choice)))} 適用。[/green]")
                    else: console.print("[yellow]適用対象なし。[/yellow]")
                elif user_choice == 'all': selected_edits_to_apply = validated_edits_with_indices; console.print("[green]全編集適用。[/green]")
                elif user_choice == 'n': console.print("[yellow]編集キャンセル。[/yellow]")
                else: is_feedback = True; feedback_text = user_choice; console.print("[yellow]フィードバック受付。[/yellow]")

                if selected_edits_to_apply:
                    try:
                        selected_edits_to_apply.sort(key=lambda item: item["start"], reverse=True)
                        new_content, applied_count = current_content, 0
                        for edit_info in selected_edits_to_apply:
                             escaped_context = re.escape(edit_info['old']); matches = list(re.finditer(escaped_context, new_content))
                             if len(matches) == 1: start, end = matches[0].span(); new_content = new_content[:start] + edit_info['new'] + new_content[end:]; applied_count += 1
                             else: console.print(f"[bold red]エラー: 編集 {edit_info['display_id']} 適用中問題発生。スキップ。[/]"); console.print(f"[dim]対象: '{edit_info['old']}', 発見数: {len(matches)}[/dim]")
                        if applied_count > 0 and perform_write(filepath, new_content, file_encoding):
                            edit_context.current_content = new_content; edit_context.last_applied_edit = agent_output
                            edit_context.last_proposal = None; edit_applied = True
                            conversation_history.append({"role": "user", "content": [{"type": "input_text", "text": f"編集 {user_choice} を承認"}]})
                        elif applied_count > 0: console.print("[red]書込失敗。[/red]")
                    except Exception as e: console.print(f"[bold red]複数箇所置換/書込エラー: {e}[/]")
                elif user_choice == 'n':
                     conversation_history.append({"role": "user", "content": [{"type": "input_text", "text": "n (拒否)"}]})
                # is_feedback の場合は次のループで処理

            # 3. 全体置換提案
            elif agent_output.status == "replace_all":
                console.print(f"\n[bold green]AI ({result.last_agent.name}) 全体書換提案:[/]")
                console.print(f"説明: {agent_output.explanation}")
                if current_content is None:
                     console.print("[bold red]エラー: 元内容なし、差分表示不可。[/bold red]")
                     console.print("[bold blue]--- 新内容 ---[/]"); syntax = Syntax(agent_output.new_content, filepath.suffix.lstrip('.'), theme="default", line_numbers=True); console.print(syntax); console.print("[bold blue]--- 新内容終 ---[/]")
                else: console.print(f"元 約{len(current_content)}文字 -> 新 約{len(agent_output.new_content)}文字"); display_diff(current_content, agent_output.new_content)

                user_choice = ask_confirmation("\nファイル全体を上書きしますか？")
                if user_choice == 'y':
                    if perform_write(filepath, agent_output.new_content, file_encoding):
                        edit_context.current_content = agent_output.new_content; edit_context.last_applied_edit = agent_output
                        edit_context.last_proposal = None; edit_applied = True
                        conversation_history.append({"role": "user", "content": [{"type": "input_text", "text": "y (承認)"}]})
                    else: console.print("[red]書込失敗。[/red]")
                elif user_choice == 'n':
                    console.print("[yellow]全体書換キャンセル。[/yellow]")
                    conversation_history.append({"role": "user", "content": [{"type": "input_text", "text": "n (拒否)"}]})
                else:
                    console.print("[yellow]フィードバック受付。[/yellow]")
                    # フィードバックは次のループで処理

            # 4. 要確認/会話/拒否
            elif agent_output.status in ["clarification_needed", "conversation", "rejected"]:
                color = "yellow" if agent_output.status == "clarification_needed" else "blue" if agent_output.status == "conversation" else "red"
                agent_name = result.last_agent.name if result.last_agent else "AI"
                console.print(f"\n[bold {color}]{agent_name}:[/][{color}] {agent_output.message}[/]")
                edit_context.last_proposal = None # 提案クリア
                edit_context.last_applied_edit = None # 編集なし
                # 会話履歴にはAI応答として既に追加済み

            # 5. その他の予期せぬステータス
            else:
                console.print(f"[bold red]エラー: AIから予期しないステータス '{getattr(agent_output, 'status', 'N/A')}'[/]")
                try: console.print(f"応答内容: {agent_output.model_dump_json(indent=2)}")
                except Exception: console.print(f"応答内容(生): {agent_output}")
                edit_context.last_proposal = None; edit_context.last_applied_edit = None

        # --- ループ内のエラーハンドリング ---
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            console.print(f"[bold red]Agent実行エラー ({type(e).__name__}): {e}[/]"); console.print("[yellow]指示変更か/reloadを。[/yellow]"); edit_context.last_proposal = None
        except BadRequestError as e:
            console.print(f"[bold red]OpenAI APIエラー: {e}[/]"); console.print("[yellow]APIキー/接続確認を。[/yellow]"); edit_context.last_proposal = None
        except AgentsException as e:
            console.print(f"[bold red]Agent SDK エラー: {e}[/]"); edit_context.last_proposal = None
        except EOFError: console.print("\n[bold magenta]入力終了。終了します。[/]"); break
        except KeyboardInterrupt: console.print("\n[bold magenta]中断。終了します。[/]"); break
        except Exception as e:
            console.print(f"[bold red]予期せぬエラー: {e}[/]"); import traceback; traceback.print_exc(); edit_context.last_proposal = None


# --- プログラムのエントリポイント ---
if __name__ == "__main__":
    try: asyncio.run(main())
    except Exception as e:
        console.print(f"\n[bold red]プログラム実行中致命的エラー: {e}[/]"); import traceback; traceback.print_exc()
