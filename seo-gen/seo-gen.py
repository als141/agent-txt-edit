# -*- coding: utf-8 -*-
import os
import json
import asyncio
import re
import time # リトライのためのtimeモジュールをインポート
from pathlib import Path
from openai import AsyncOpenAI, BadRequestError, InternalServerError # InternalServerErrorをインポート
from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent # Streaming用イベント
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.syntax import Syntax
from typing import List, Dict, Union, Optional, Tuple, Any, Literal, Callable, Awaitable
from pydantic import BaseModel, Field, ValidationError, field_validator
from dataclasses import dataclass, field
import uuid
import traceback # エラー詳細表示用

# --- Agents SDK ---
from agents import (
    Agent,
    Runner,
    RunConfig,
    function_tool,
    ModelSettings, # ModelSettings をインポート
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
    # ツール
    WebSearchTool,
    FileSearchTool, # ベクトルストア検索用
    # モデル
    Model,
    OpenAIResponsesModel, # デフォルト
    OpenAIChatCompletionsModel, # Chat Completions API用
    ItemHelpers,
    # Streaming用
    StreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    AgentUpdatedStreamEvent,
    # MessageInputItem と InputTextContentItem のインポートを削除
)
# LiteLLM 連携 (オプション)
try:
    from agents.extensions.models.litellm_model import LitellmModel
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    LitellmModel = None # type: ignore

# MCP用クラス (今回は直接使用しないが、参考としてコメントアウト)
# from agents.mcp.server import MCPServerStdio, MCPServer
# from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
# --------------------

# --- 初期設定 ---
console = Console()
load_dotenv()

# APIキー設定 (環境変数から読み込み)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 他のLLMプロバイダーのキーも必要に応じて設定
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    console.print("[bold red]エラー: OPENAI_API_KEY が .env ファイルに設定されていません。[/bold red]")
    # 必要に応じてプログラムを終了させるか、デフォルトキーを設定
    exit() # APIキーがないと動作しないため終了

# デフォルトのOpenAIクライアントとモデル
# 必要に応じて set_default_openai_client や set_default_openai_api で変更可能
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
# モデル名をgpt-4.1-miniに変更
DEFAULT_MODEL = "gpt-4.1"
RESEARCH_MODEL = "gpt-4.1" # リサーチもminiで試す
WRITING_MODEL = "gpt-4.1"  # 執筆もminiで試す (o4-miniは存在しない可能性)
EDITING_MODEL = "gpt-4.1"  # 編集もminiで試す

# リトライ設定
MAX_RETRIES = 3 # 最大リトライ回数
INITIAL_RETRY_DELAY = 1 # 初期リトライ遅延（秒）

# --- Pydanticモデル定義 (Agentの出力型) ---
# ArticleSection のみ変更 (SectionWriterAgentは直接これを返さなくなる)
class ThemeIdea(BaseModel):
    """単一のテーマ案"""
    title: str = Field(description="記事のタイトル案")
    description: str = Field(description="テーマの簡単な説明とSEO的な狙い")
    keywords: List[str] = Field(description="関連するSEOキーワード")

class ThemeProposal(BaseModel):
    """テーマ提案のリスト"""
    status: Literal["theme_proposal"] = Field(description="出力タイプ: テーマ提案")
    themes: List[ThemeIdea] = Field(description="提案するテーマのリスト")

class OutlineSection(BaseModel):
    """アウトラインの単一セクション（見出し）"""
    heading: str = Field(description="セクションの見出し (例: H2, H3)")
    estimated_chars: Optional[int] = Field(default=None, description="このセクションの推定文字数")
    subsections: Optional[List['OutlineSection']] = Field(default=None, description="サブセクションのリスト（ネスト構造）")

class Outline(BaseModel):
    """記事のアウトライン"""
    status: Literal["outline"] = Field(description="出力タイプ: アウトライン")
    title: str = Field(description="記事の最終タイトル")
    suggested_tone: str = Field(description="提案する記事のトーン（例: 丁寧な解説調、フレンドリー、専門的）")
    sections: List[OutlineSection] = Field(description="記事のセクション（見出し）リスト")

# ArticleSection: SectionWriterAgent はこれを直接返さなくなるが、
# メインループで生成されたHTMLからこの構造を組み立てるために定義は残す
class ArticleSection(BaseModel):
    """生成された記事の単一セクション (メインループで構築)"""
    status: Literal["article_section"] = Field(default="article_section", description="出力タイプ: 記事セクション")
    section_index: int = Field(description="生成対象のセクションインデックス（Outline.sectionsのインデックス、0ベース）")
    heading: str = Field(description="生成されたセクションの見出し")
    html_content: str = Field(description="生成されたセクションのHTMLコンテンツ")

class RevisedArticle(BaseModel):
    """推敲・編集後の完成記事"""
    status: Literal["revised_article"] = Field(description="出力タイプ: 完成記事")
    title: str = Field(description="最終的な記事タイトル")
    final_html_content: str = Field(description="推敲・編集後の完全なHTMLコンテンツ")

class ClarificationNeeded(BaseModel):
    """ユーザーへの確認・質問"""
    status: Literal["clarification_needed"] = Field(description="出力タイプ: 要確認")
    message: str = Field(description="ユーザーへの具体的な質問や確認事項")

class StatusUpdate(BaseModel):
    """処理状況のアップデート"""
    status: Literal["status_update"] = Field(description="出力タイプ: 状況更新")
    message: str = Field(description="現在の処理状況や次のステップに関するメッセージ")

# --- リサーチ関連モデル (強化版) ---
class ResearchQuery(BaseModel):
    """リサーチプラン内の単一検索クエリ"""
    query: str = Field(description="実行する具体的な検索クエリ")
    focus: str = Field(description="このクエリで特に調査したい点")

class ResearchPlan(BaseModel):
    """リサーチ計画"""
    status: Literal["research_plan"] = Field(description="出力タイプ: リサーチ計画")
    topic: str = Field(description="リサーチ対象のトピック（記事テーマ）")
    queries: List[ResearchQuery] = Field(description="実行する検索クエリのリスト")

class SourceSnippet(BaseModel): # 新しいモデル: 詳細な情報と出典を紐付ける
    """リサーチ結果からの詳細な抜粋と出典情報"""
    snippet_text: str = Field(description="記事作成に役立つ具体的な情報やデータの抜粋")
    source_url: str = Field(description="この抜粋の出典元URL（可能な限り、最も具体的なページ）")
    source_title: Optional[str] = Field(default=None, description="出典元ページのタイトル")

class ResearchQueryResult(BaseModel): # 修正: より詳細な情報を保持
    """単一クエリのリサーチ結果（詳細版）"""
    status: Literal["research_query_result"] = Field(description="出力タイプ: リサーチクエリ結果")
    query: str = Field(description="実行された検索クエリ")
    summary: str = Field(description="検索結果の主要な情報の要約（簡潔に）")
    detailed_findings: List[SourceSnippet] = Field(description="記事作成に役立つ詳細な情報抜粋と出典URLのリスト")
    # source_urls は detailed_findings に含まれるため削除

class KeyPoint(BaseModel): # 新しいモデル: キーポイントと出典を紐付ける
    """リサーチレポートのキーポイントと関連情報源"""
    point: str = Field(description="記事に含めるべき重要なポイントや事実")
    supporting_sources: List[str] = Field(description="このポイントを裏付ける情報源URLのリスト")

class ResearchReport(BaseModel): # 修正: 詳細な情報と出典を保持
    """リサーチ結果の要約レポート（詳細版）"""
    status: Literal["research_report"] = Field(description="出力タイプ: リサーチレポート")
    topic: str = Field(description="リサーチ対象のトピック")
    overall_summary: str = Field(description="リサーチ全体から得られた主要な洞察やポイントの要約")
    key_points: List[KeyPoint] = Field(description="記事に含めるべき重要なポイントや事実と、その情報源リスト")
    interesting_angles: List[str] = Field(description="記事を面白くするための切り口や視点のアイデア")
    all_sources: List[str] = Field(description="参照した全ての情報源URLのリスト（重複削除済み、重要度順推奨）")

# エージェントが出力しうる型のUnion (ArticleSection を削除)
AgentOutput = Union[
    ThemeProposal, Outline, RevisedArticle, ClarificationNeeded, StatusUpdate,
    ResearchPlan, ResearchQueryResult, ResearchReport
]

# --- コンテキストクラス ---
@dataclass
class ArticleContext:
    """記事生成プロセス全体で共有されるコンテキスト"""
    # --- ユーザー入力 ---
    initial_keywords: List[str] = field(default_factory=list)
    target_persona: Optional[str] = None
    target_length: Optional[int] = None # 目標文字数
    num_theme_proposals: int = 3
    vector_store_id: Optional[str] = None # File Search用
    num_research_queries: int = 5 # リサーチクエリ数の上限

    # --- 企業情報 (ツールで取得想定) ---
    company_name: Optional[str] = None
    company_description: Optional[str] = None
    company_style_guide: Optional[str] = None # 文体、トンマナなど
    past_articles_summary: Optional[str] = None # 過去記事の傾向

    # --- 生成プロセス状態 ---
    current_step: Literal[
        "start", "theme_proposed", "theme_selected",
        "research_planning", "research_plan_generated", "researching", "research_synthesizing", "research_report_generated", # リサーチステップ追加
        "outline_generation", # ステップ名変更
        "outline_generated", "writing_sections", "editing", "completed", "error"
    ] = "start"
    selected_theme: Optional[ThemeIdea] = None
    research_plan: Optional[ResearchPlan] = None # リサーチプラン
    current_research_query_index: int = 0 # 現在のリサーチクエリインデックス
    research_query_results: List[ResearchQueryResult] = field(default_factory=list) # 修正: クエリ結果を保存 (型変更)
    research_report: Optional[ResearchReport] = None # 修正: 最終リサーチレポート (型変更)
    generated_outline: Optional[Outline] = None
    current_section_index: int = 0 # 執筆対象のセクションインデックス (0ベース)
    generated_sections_html: List[str] = field(default_factory=list) # 各セクションのHTMLを格納
    full_draft_html: Optional[str] = None # 結合後のドラフト
    final_article_html: Optional[str] = None # 最終成果物
    error_message: Optional[str] = None
    # last_agent_output は AgentOutput または ArticleSection を保持するように変更
    last_agent_output: Optional[Union[AgentOutput, ArticleSection]] = None
    # section_writer_history: List[MessageInputItem] = field(default_factory=list) # 修正: 型ヒントを変更
    section_writer_history: List[Dict[str, Any]] = field(default_factory=list) # 修正: より汎用的な型ヒントを使用

    def get_full_draft(self) -> str:
        """生成されたセクションを結合して完全なドラフトHTMLを返す"""
        return "\n".join(self.generated_sections_html)

    def add_query_result(self, result: ResearchQueryResult): # 修正: 型ヒント変更
        """リサーチクエリ結果を追加"""
        self.research_query_results.append(result)

    def clear_section_writer_history(self):
        """セクションライターの履歴をクリア"""
        self.section_writer_history = []

    # 修正: 会話履歴にメッセージを追加するヘルパーメソッド
    def add_to_section_writer_history(self, role: Literal["user", "assistant", "system", "developer", "tool"], content: str):
        """指定されたロールと内容でメッセージを会話履歴に追加する"""
        # 修正: roleに応じてcontentのtypeを変更
        content_type = "output_text" if role == "assistant" else "input_text"
        # システムメッセージの場合も input_text とする (SDKの挙動に合わせる)
        if role == "system" or role == "developer":
             content_type = "input_text"

        message: Dict[str, Any] = {
            "role": role,
            "content": [{"type": content_type, "text": content}]
        }
        self.section_writer_history.append(message)


# --- ツール定義 ---
# Web検索ツール (Agents SDK標準) - ResearcherAgentが使用
web_search_tool = WebSearchTool(
    user_location={"type": "approximate", "country": "JP"}
)

# ファイル検索ツール (Agents SDK標準) - 必要に応じて使用
# file_search_tool = FileSearchTool(vector_store_ids=[...]) if context.vector_store_id else None

# 会社情報取得ツール (ダミー)
@function_tool
async def get_company_data(ctx: RunContextWrapper[ArticleContext]) -> Dict[str, Any]:
    """
    顧客企業のデータベースやCMSから関連情報を取得します。
    (この実装はダミーです。実際のシステムではAPI呼び出し等に置き換えてください)
    """
    console.print("[dim]ツール実行(get_company_data): ダミーデータを返します。[/dim]")
    return {
        "success": True,
        "company_name": ctx.context.company_name or "株式会社ジョンソンホームズ",
        "company_description": ctx.context.company_description or "住宅の設計・施工、リフォーム工事の設計・施工、不動産の売買および斡旋、インテリア商品の販売、オーダーソファの製造・販売、レストラン・カフェ運営、保険事業、住宅FC本部",
        "company_style_guide": ctx.context.company_style_guide or "文体は丁寧語（ですます調）を基本とし、専門用語は避ける。読者に寄り添うフレンドリーなトーン。",
        "past_articles_summary": ctx.context.past_articles_summary or "過去にはブログやコラム系の記事が多い。",
    }

# 競合分析ツール (ダミー)
@function_tool
async def analyze_competitors(ctx: RunContextWrapper[ArticleContext], query: str) -> Dict[str, Any]:
    """
    指定されたクエリでWeb検索を行い、競合となる記事の傾向を分析します。
    (この実装はダミーです。WebSearchToolの結果を解析する処理に置き換えてください)

    Args:
        query: 競合分析のための検索クエリ（例：「芝生 育て方 ガイド」）
    """
    console.print(f"[dim]ツール実行(analyze_competitors): クエリ '{query}' のダミー分析結果を返します。[/dim]")
    # ダミーデータに少し具体性を持たせる
    common_sections_map = {
        "芝生 育て方 初心者": ["準備するもの", "種まき", "水やり", "肥料", "芝刈り"],
        "芝生 手入れ コツ": ["サッチング", "エアレーション", "目土入れ", "病害虫対策"],
    }
    return {
        "success": True,
        "summary": f"'{query}' に関する競合記事は、主に基本的な手入れ方法や季節ごとの注意点を解説しています。",
        "common_sections": common_sections_map.get(query, ["基本的な手入れ", "季節のケア", "トラブルシューティング"]),
        "estimated_length_range": "1500〜3000文字",
    }

# --- 動的プロンプト生成関数 ---
# (変更なしの部分は省略)
def create_theme_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        company_info_str = f"企業名: {ctx.context.company_name}\n概要: {ctx.context.company_description}\n文体ガイド: {ctx.context.company_style_guide}\n過去記事傾向: {ctx.context.past_articles_summary}" if ctx.context.company_name else "企業情報なし"
        full_prompt = f"""{base_prompt}

--- 入力情報 ---
キーワード: {', '.join(ctx.context.initial_keywords)}
ターゲットペルソナ: {ctx.context.target_persona or '指定なし'}
提案するテーマ数: {ctx.context.num_theme_proposals}
企業情報:\n{company_info_str}
---

あなたの応答は必ず `ThemeProposal` または `ClarificationNeeded` 型のJSON形式で出力してください。
"""
        return full_prompt
    return dynamic_instructions_func

def create_research_planner_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.selected_theme:
            return "エラー: リサーチ計画を作成するためのテーマが選択されていません。"

        full_prompt = f"""{base_prompt}

--- リサーチ対象テーマ ---
タイトル: {ctx.context.selected_theme.title}
説明: {ctx.context.selected_theme.description}
キーワード: {', '.join(ctx.context.selected_theme.keywords)}
ターゲットペルソナ: {ctx.context.target_persona or '指定なし'}
---

**重要:**
- 上記テーマについて深く掘り下げるための、具体的で多様な検索クエリを **{ctx.context.num_research_queries}個** 生成してください。
- 各クエリには、そのクエリで何を明らかにしたいか（focus）を明確に記述してください。
- あなたの応答は必ず `ResearchPlan` 型のJSON形式で出力してください。
"""
        return full_prompt
    return dynamic_instructions_func

# 修正: ResearcherAgent のプロンプト (詳細な情報収集を指示)
def create_researcher_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.research_plan or ctx.context.current_research_query_index >= len(ctx.context.research_plan.queries):
            return "エラー: 有効なリサーチプランまたは実行すべきクエリがありません。"

        current_query = ctx.context.research_plan.queries[ctx.context.current_research_query_index]

        full_prompt = f"""{base_prompt}

--- 現在のリサーチタスク ---
記事テーマ: {ctx.context.research_plan.topic}
今回の検索クエリ: "{current_query.query}"
このクエリの焦点: {current_query.focus}
---

**重要:**
- 上記の検索クエリを使用して `web_search` ツールを実行してください。
- 検索結果を**深く分析**し、記事テーマとクエリの焦点に関連する**具体的な情報、データ、主張、引用**などを**詳細に抽出**してください。
- 抽出した各情報について、**最も信頼性が高く具体的な出典元URLとそのタイトル**を特定し、`SourceSnippet` 形式でリスト化してください。単なる検索結果一覧のURLではなく、情報が実際に記載されているページのURLを重視してください。公式HPや信頼できる情報源を優先してください。
- 検索結果全体の**簡潔な要約 (summary)** も生成してください。
- あなたの応答は必ず `ResearchQueryResult` 型のJSON形式で出力してください。他のテキストは一切含めないでください。
- **`save_research_snippet` ツールは使用しないでください。**
"""
        return full_prompt
    return dynamic_instructions_func

# 修正: ResearchSynthesizerAgent のプロンプト (詳細なレポート作成を指示)
def create_research_synthesizer_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.research_query_results:
            return "エラー: 要約するためのリサーチ結果がありません。"

        results_str = ""
        all_sources_set = set() # 重複削除用
        for i, result in enumerate(ctx.context.research_query_results):
            results_str += f"--- クエリ結果 {i+1} ({result.query}) ---\n"
            results_str += f"要約: {result.summary}\n"
            results_str += "詳細な発見:\n"
            for finding in result.detailed_findings:
                results_str += f"- 抜粋: {finding.snippet_text}\n"
                results_str += f"  出典: [{finding.source_title or finding.source_url}]({finding.source_url})\n"
                all_sources_set.add(finding.source_url) # URLをセットに追加
            results_str += "\n"

        all_sources_list = sorted(list(all_sources_set)) # 重複削除してリスト化

        full_prompt = f"""{base_prompt}

--- リサーチ対象テーマ ---
{ctx.context.selected_theme.title if ctx.context.selected_theme else 'N/A'}

--- 収集されたリサーチ結果 (詳細) ---
{results_str[:15000]}
{ "... (以下省略)" if len(results_str) > 15000 else "" }
---

**重要:**
- 上記の詳細なリサーチ結果全体を分析し、記事執筆に役立つように情報を統合・要約してください。
- 以下の要素を含む**実用的で詳細なリサーチレポート**を作成してください:
    - `overall_summary`: リサーチ全体から得られた主要な洞察やポイントの要約。
    - `key_points`: 記事に含めるべき重要なポイントや事実をリスト形式で記述し、各ポイントについて**それを裏付ける情報源URL (`supporting_sources`)** を `KeyPoint` 形式で明確に紐付けてください。
    - `interesting_angles`: 記事を面白くするための切り口や視点のアイデアのリスト形式。
    - `all_sources`: 参照した全ての情報源URLのリスト（重複削除済み、可能であれば重要度順）。
- レポートは論文調ではなく、記事作成者がすぐに使えるような分かりやすい言葉で記述してください。
- あなたの応答は必ず `ResearchReport` 型のJSON形式で出力してください。
"""
        return full_prompt
    return dynamic_instructions_func

# 修正: OutlineAgent のプロンプト (詳細なリサーチレポートを参照)
def create_outline_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.selected_theme or not ctx.context.research_report:
            return "エラー: テーマまたはリサーチレポートが利用できません。"

        company_info_str = f"文体ガイド: {ctx.context.company_style_guide}" if ctx.context.company_style_guide else "企業文体ガイドなし"
        # リサーチレポートのキーポイントを整形
        research_key_points_str = ""
        for kp in ctx.context.research_report.key_points:
            sources_str = ", ".join(kp.supporting_sources[:2]) # 代表的なソースをいくつか表示
            if len(kp.supporting_sources) > 2: sources_str += ", ..."
            research_key_points_str += f"- {kp.point} (出典: {sources_str})\n"

        research_summary = f"リサーチ要約: {ctx.context.research_report.overall_summary}\n主要ポイント:\n{research_key_points_str}面白い切り口: {', '.join(ctx.context.research_report.interesting_angles)}"

        full_prompt = f"""{base_prompt}

--- 入力情報 ---
選択されたテーマ:
  タイトル: {ctx.context.selected_theme.title}
  説明: {ctx.context.selected_theme.description}
  キーワード: {', '.join(ctx.context.selected_theme.keywords)}
ターゲット文字数: {ctx.context.target_length or '指定なし（標準的な長さで）'}
ターゲットペルソナ: {ctx.context.target_persona or '指定なし'}
{company_info_str}
--- 詳細なリサーチ結果 ---
{research_summary}
参照した全情報源URL数: {len(ctx.context.research_report.all_sources)}
---

**重要:**
- 上記のテーマと**詳細なリサーチ結果**、そして競合分析の結果（ツール使用）に基づいて、記事のアウトラインを作成してください。
- リサーチ結果の**キーポイント（出典情報も考慮）**や面白い切り口をアウトラインに反映させてください。
- **ターゲットペルソナ（{ctx.context.target_persona or '指定なし'}）** が読みやすいように、日本の一般的なブログやコラムのような、**親しみやすく分かりやすいトーン**でアウトラインを作成してください。記事全体のトーンも提案してください。
- あなたの応答は必ず `Outline` または `ClarificationNeeded` 型のJSON形式で出力してください。
- 文字数指定がある場合は、それに応じてセクション数や深さを調整してください。
"""
        return full_prompt
    return dynamic_instructions_func

# 修正: Section Writer のプロンプト (詳細なリサーチレポート参照とリンク生成指示)
def create_section_writer_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.generated_outline or ctx.context.current_section_index >= len(ctx.context.generated_outline.sections):
            return "エラー: 有効なアウトラインまたはセクションインデックスがありません。"
        if not ctx.context.research_report:
            return "エラー: 参照すべきリサーチレポートがありません。"

        target_section = ctx.context.generated_outline.sections[ctx.context.current_section_index]
        target_index = ctx.context.current_section_index # ターゲットインデックスを明確に変数化 (0ベース)
        target_heading = target_section.heading
        target_persona = ctx.context.target_persona or '指定なし' # ペルソナを明記

        section_target_chars = None
        if ctx.context.target_length and len(ctx.context.generated_outline.sections) > 0:
            total_sections = len(ctx.context.generated_outline.sections)
            estimated_total_body_chars = ctx.context.target_length * 0.8
            section_target_chars = int(estimated_total_body_chars / total_sections)

        outline_context = "\n".join([f"{i+1}. {s.heading}" for i, s in enumerate(ctx.context.generated_outline.sections)])

        # リサーチレポートのキーポイントと出典情報を整形してコンテキストに追加
        research_context_str = f"リサーチ要約: {ctx.context.research_report.overall_summary[:500]}...\n"
        research_context_str += "主要なキーポイントと出典:\n"
        for kp in ctx.context.research_report.key_points:
            sources_str = ", ".join([f"[{url.split('/')[-1] if url.split('/')[-1] else url}]({url})" for url in kp.supporting_sources]) # URLからファイル名等を取得して表示
            research_context_str += f"- {kp.point} (出典: {sources_str})\n"
        research_context_str += f"参照した全情報源URL数: {len(ctx.context.research_report.all_sources)}\n"

        company_style_guide = ctx.context.company_style_guide or '指定なし' # スタイルガイドを明記

        # プロンプトに渡す情報を整理
        full_prompt = f"""{base_prompt}

--- 記事全体の情報 ---
記事タイトル: {ctx.context.generated_outline.title}
記事全体のキーワード: {', '.join(ctx.context.selected_theme.keywords) if ctx.context.selected_theme else 'N/A'}
記事全体のトーン: {ctx.context.generated_outline.suggested_tone}
ターゲットペルソナ: {target_persona}
企業スタイルガイド: {company_style_guide}
記事のアウトライン（全体像）:
{outline_context}
--- 詳細なリサーチ情報 ---
{research_context_str[:10000]}
{ "... (以下省略)" if len(research_context_str) > 10000 else "" }
---

--- **あなたの現在のタスク** ---
あなたは **セクションインデックス {target_index}**、見出し「**{target_heading}**」の内容をHTML形式で執筆するタスク**のみ**を担当します。
このセクションの目標文字数: {section_target_chars or '指定なし（流れに合わせて適切に）'}
---

--- **【最重要】執筆スタイルとトーンについて** ---
あなたの役割は、単に情報をHTMLにするだけでなく、**まるで経験豊富な友人が「{target_persona}」に語りかけるように**、親しみやすく、分かりやすい文章でセクションを執筆することです。
- **日本の一般的なブログ記事やコラムのような、自然で人間味あふれる、温かいトーン**を心がけてください。堅苦しい表現や機械的な言い回しは避けてください。
- 読者に直接語りかけるような表現（例：「〜だと思いませんか？」「まずは〜から始めてみましょう！」「〜なんてこともありますよね」）や、共感を誘うような言葉遣いを積極的に使用してください。
- 専門用語は避け、どうしても必要な場合は簡単な言葉で補足説明を加えてください。箇条書きなども活用し、情報を整理して伝えると良いでしょう。
- 可能であれば、具体的な体験談（想像でも構いません）や、読者が抱きそうな疑問に答えるような形で内容を構成すると、より読者の心に響きます。
- 企業スタイルガイド「{company_style_guide}」も必ず遵守してください。
---

--- 執筆ルール ---
1.  **提供される会話履歴（直前のセクションの内容など）と、上記「詳細なリサーチ情報」を十分に考慮し、** 前のセクションから自然につながるように、かつ、このセクション（インデックス {target_index}、見出し「{target_heading}」）の主題に沿った文章を作成してください。
2.  **リサーチ情報で示された事実やデータに基づいて執筆し、必要に応じて、信頼できる情報源（特に公式HPなど）へのHTMLリンク (`<a href="URL">リンクテキスト</a>`) を自然な形で含めてください。** リンクテキストは具体的に、例えば会社名やサービス名、情報の内容を示すものにしてください。ただし、過剰なリンクやSEOに不自然なリンクは避けてください。リサーチ情報に記載のない情報は含めないでください。
3.  他のセクションの内容は絶対に生成しないでください。
4.  必ず `<p>`, `<h2>`, `<h3>`, `<ul>`, `<li>`, `<strong>`, `<em>`, `<a>` などの基本的なHTMLタグを使用し、構造化されたコンテンツを生成してください。`<h2>` タグはこのセクションの見出し「{target_heading}」にのみ使用してください。
5.  SEOを意識し、記事全体のキーワードやこのセクションに関連するキーワードを**自然に**含めてください。（ただし、自然さを損なうような無理なキーワードの詰め込みは避けてください）
6.  上記の【執筆スタイルとトーンについて】の指示に従い、創造性を発揮し、読者にとって価値のあるオリジナルな文章を作成してください。
---

--- **【最重要】出力形式について** ---
あなたの応答は**必ず**、指示されたセクション（インデックス {target_index}、見出し「{target_heading}」）の**HTMLコンテンツ文字列のみ**を出力してください。
- **JSON形式や ```html のようなマークダウン形式は絶対に使用しないでください。**
- **「はい、以下にHTMLを記述します」のような前置きや、説明文、コメントなども一切含めないでください。**
- **出力は `<h2...>` または `<p...>` タグから始まり、そのセクションの最後のHTMLタグで終わるようにしてください。**
- **指定されたセクションのHTMLコンテンツだけを、そのまま出力してください。**
"""
        return full_prompt
    return dynamic_instructions_func


# 修正: EditorAgent のプロンプト (詳細なリサーチレポート参照とリンク確認)
def create_editor_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.full_draft_html:
            return "エラー: 編集対象のドラフト記事がありません。"
        if not ctx.context.research_report:
            return "エラー: 参照すべきリサーチレポートがありません。"

        # リサーチレポートのキーポイントと出典情報を整形
        research_context_str = f"リサーチ要約: {ctx.context.research_report.overall_summary[:500]}...\n"
        research_context_str += "主要なキーポイントと出典:\n"
        for kp in ctx.context.research_report.key_points:
            sources_str = ", ".join([f"[{url.split('/')[-1] if url.split('/')[-1] else url}]({url})" for url in kp.supporting_sources])
            research_context_str += f"- {kp.point} (出典: {sources_str})\n"
        research_context_str += f"参照した全情報源URL数: {len(ctx.context.research_report.all_sources)}\n"


        full_prompt = f"""{base_prompt}

--- 編集対象記事ドラフト (HTML) ---
```html
{ctx.context.full_draft_html[:15000]}
{ "... (以下省略)" if len(ctx.context.full_draft_html) > 15000 else "" }
```
---

--- 記事の要件 ---
タイトル: {ctx.context.generated_outline.title if ctx.context.generated_outline else 'N/A'}
キーワード: {', '.join(ctx.context.selected_theme.keywords) if ctx.context.selected_theme else 'N/A'}
ターゲットペルソナ: {ctx.context.target_persona or '指定なし'}
目標文字数: {ctx.context.target_length or '指定なし'}
トーン: {ctx.context.generated_outline.suggested_tone if ctx.context.generated_outline else 'N/A'}
企業スタイルガイド: {ctx.context.company_style_guide or '指定なし'}
--- 詳細なリサーチ情報 ---
{research_context_str[:10000]}
{ "... (以下省略)" if len(research_context_str) > 10000 else "" }
---

**重要:**
- 上記のドラフトHTMLをレビューし、記事の要件と**詳細なリサーチ情報**に基づいて推敲・編集してください。
- **特に、文章全体がターゲットペルソナ（{ctx.context.target_persona or '指定なし'}）にとって自然で、親しみやすく、分かりやすい言葉遣いになっているか** を重点的に確認してください。機械的な表現や硬い言い回しがあれば、より人間味のある表現に修正してください。
- チェックポイント:
    - 全体の流れと一貫性
    - 各セクションの内容の質と正確性 (**リサーチ情報との整合性、事実確認**)
    - 文法、スペル、誤字脱字
    - 指示されたトーンとスタイルガイドの遵守 (**自然さ、親しみやすさ重視**)
    - ターゲットペルソナへの適合性
    - SEO最適化（キーワードの自然な使用、見出し構造）
    - **含まれているHTMLリンク (`<a>` タグ) がリサーチ情報に基づいており、適切かつ自然に使用されているか。リンク切れや不適切なリンクがないか。**
    - 人間らしい自然な文章表現、独創性
    - HTML構造の妥当性
- 必要な修正を直接HTMLに加えてください。
- あなたの応答は必ず `RevisedArticle` 型のJSON形式で、`final_html_content` に編集後の完全なHTML文字列を入れて出力してください。

"""
        return full_prompt
    return dynamic_instructions_func


# --- エージェント定義 ---

# 1. テーマ提案エージェント
THEME_AGENT_BASE_PROMPT = """
あなたはSEO記事のテーマを考案する専門家です。
与えられたキーワード、ターゲットペルソナ、企業情報を分析し、読者の検索意図とSEO効果を考慮した上で、創造的で魅力的な記事テーマ案を複数生成します。
必要であれば `get_company_data` ツールで企業情報を補強し、`web_search` ツールで関連トレンドや競合を調査できます。
情報が不足している場合は、ユーザーに質問してください。
"""
theme_agent = Agent[ArticleContext](
    name="ThemeAgent",
    instructions=create_theme_instructions(THEME_AGENT_BASE_PROMPT),
    model=DEFAULT_MODEL,
    tools=[get_company_data, web_search_tool],
    output_type=AgentOutput, # ThemeProposal or ClarificationNeeded
)

# --- リサーチエージェント群 ---
# 2. リサーチプランナーエージェント
RESEARCH_PLANNER_AGENT_BASE_PROMPT = """
あなたは優秀なリサーチプランナーです。
与えられた記事テーマに基づき、そのテーマを深く掘り下げ、読者が知りたいであろう情報を網羅するための効果的なWeb検索クエリプランを作成します。
"""
research_planner_agent = Agent[ArticleContext](
    name="ResearchPlannerAgent",
    instructions=create_research_planner_instructions(RESEARCH_PLANNER_AGENT_BASE_PROMPT),
    model=RESEARCH_MODEL,
    tools=[], # 基本的にツールは不要
    output_type=AgentOutput, # ResearchPlan or ClarificationNeeded
)

# 3. リサーチャーエージェント (修正済み: 詳細情報収集)
RESEARCHER_AGENT_BASE_PROMPT = """
あなたは熟練したディープリサーチャーです。
指定された検索クエリでWeb検索を実行し、結果を**深く分析**します。
記事テーマに関連する**具体的で信頼できる情報、データ、主張、引用**を**詳細に抽出し、最も適切な出典元URLとタイトルを特定**して、指定された形式で返します。
**必ず web_search ツールを使用してください。**
"""
researcher_agent = Agent[ArticleContext](
    name="ResearcherAgent",
    instructions=create_researcher_instructions(RESEARCHER_AGENT_BASE_PROMPT), # 修正された関数を使用
    model=RESEARCH_MODEL,
    tools=[web_search_tool], # save_research_snippet を削除済み
    output_type=ResearchQueryResult, # 修正された ResearchQueryResult を返す
)

# 4. リサーチシンセサイザーエージェント (修正済み: 詳細レポート作成)
RESEARCH_SYNTHESIZER_AGENT_BASE_PROMPT = """
あなたは情報を整理し、要点を抽出し、統合する専門家です。
収集された**詳細なリサーチ結果（抜粋と出典）**を分析し、記事のテーマに沿って統合・要約します。
各キーポイントについて、**それを裏付ける情報源URLを明確に紐付け**、記事作成者がすぐに活用できる**実用的で詳細なリサーチレポート**を作成します。
"""
research_synthesizer_agent = Agent[ArticleContext](
    name="ResearchSynthesizerAgent",
    instructions=create_research_synthesizer_instructions(RESEARCH_SYNTHESIZER_AGENT_BASE_PROMPT), # 修正された関数を使用
    model=RESEARCH_MODEL,
    tools=[], # 基本的にツールは不要
    output_type=AgentOutput, # 修正された ResearchReport を返す
)

# --- 記事作成エージェント群 ---
# 5. アウトライン作成エージェント (修正済み: 詳細レポート参照)
OUTLINE_AGENT_BASE_PROMPT = """
あなたはSEO記事のアウトライン（構成案）を作成する専門家です。
選択されたテーマ、目標文字数、企業のスタイルガイド、ターゲットペルソナ、そして**詳細なリサーチレポート（キーポイントと出典情報を含む）**に基づいて、論理的で網羅的、かつ読者の興味を引く記事のアウトラインを生成します。
`analyze_competitors` ツールで競合記事の構成を調査し、差別化できる構成を考案します。
`get_company_data` ツールでスタイルガイドを確認します。
文字数指定に応じて、見出しの数や階層構造を適切に調整します。
**ターゲットペルソナが読みやすいように、親しみやすく分かりやすいトーン**で記事全体のトーンも提案してください。
"""
outline_agent = Agent[ArticleContext](
    name="OutlineAgent",
    instructions=create_outline_instructions(OUTLINE_AGENT_BASE_PROMPT), # 修正された関数を使用
    model=WRITING_MODEL,
    tools=[analyze_competitors, get_company_data],
    output_type=AgentOutput, # Outline or ClarificationNeeded
)

# 6. セクション執筆エージェント (修正済み: 詳細レポート参照とリンク生成)
SECTION_WRITER_AGENT_BASE_PROMPT = """
あなたは指定された記事のセクション（見出し）に関する内容を執筆するプロのライターです。
**あなたの役割は、日本の一般的なブログやコラムのように、自然で人間味あふれる、親しみやすい文章で**、割り当てられた特定のセクションの内容をHTML形式で執筆することです。
記事全体のテーマ、アウトライン、キーワード、トーン、**会話履歴（前のセクションを含む完全な文脈）**、そして**詳細なリサーチレポート（出典情報付き）**に基づき、創造的かつSEOを意識して執筆してください。
**リサーチ情報に基づき、必要に応じて信頼できる情報源へのHTMLリンクを自然に含めてください。**
必要に応じて `web_search` ツールで最新情報や詳細情報を調査し、内容を充実させます。
**あなたのタスクは、指示された1つのセクションのHTMLコンテンツを生成することだけです。** 読者を引きつけ、価値を提供するオリジナルな文章を作成してください。
"""
section_writer_agent = Agent[ArticleContext](
    name="SectionWriterAgent",
    instructions=create_section_writer_instructions(SECTION_WRITER_AGENT_BASE_PROMPT), # 修正された関数を使用
    model=WRITING_MODEL,
    tools=[web_search_tool], # 必要に応じてWeb検索を有効化
    # output_type を削除 (構造化出力を強制しない)
)

# 7. 推敲・編集エージェント (修正済み: 詳細レポート参照とリンク確認)
EDITOR_AGENT_BASE_PROMPT = """
あなたはプロの編集者兼SEOスペシャリストです。
与えられた記事ドラフト（HTML形式）を、記事の要件（テーマ、キーワード、ペルソナ、文字数、トーン、スタイルガイド）と**詳細なリサーチレポート（出典情報付き）**を照らし合わせながら、徹底的にレビューし、推敲・編集します。
**特に、文章全体がターゲットペルソナにとって自然で、親しみやすく、分かりやすい言葉遣いになっているか** を重点的に確認し、機械的な表現があれば人間味のある表現に修正してください。
**リサーチ情報との整合性、事実確認、含まれるHTMLリンクの適切性**も厳しくチェックします。
文章の流れ、一貫性、正確性、文法、読みやすさ、独創性、そしてSEO最適化の観点から、最高品質の記事に仕上げることを目指します。
必要であれば `web_search` ツールでファクトチェックや追加情報を調査します。
最終的な成果物として、編集済みの完全なHTMLコンテンツを出力します。
"""
editor_agent = Agent[ArticleContext](
    name="EditorAgent",
    instructions=create_editor_instructions(EDITOR_AGENT_BASE_PROMPT), # 修正された関数を使用
    model=EDITING_MODEL,
    tools=[web_search_tool],
    output_type=AgentOutput, # RevisedArticle
)

# --- LiteLLM 設定例 ---
# (変更なし)
def get_litellm_agent(agent_type: Literal["editor", "writer", "researcher"], model_name: str, api_key: Optional[str] = None) -> Optional[Agent]:
    """LiteLLMを使用して指定されたタイプのエージェントを生成するヘルパー関数"""
    if not LITELLM_AVAILABLE or not LitellmModel:
        console.print("[yellow]警告: LiteLLM が利用できないため、LiteLLMモデルは使用できません。[/yellow]")
        return None

    try:
        litellm_model_instance = LitellmModel(model=model_name, api_key=api_key)
        agent_name = f"{agent_type.capitalize()}Agent_{model_name.replace('/', '_')}"
        base_prompt = ""
        tools = []
        output_type: Any = AgentOutput # デフォルトはJSON期待
        model_settings = None # LiteLLM用の設定は別途必要か確認

        if agent_type == "editor":
            base_prompt = EDITOR_AGENT_BASE_PROMPT
            instructions_func = create_editor_instructions
            tools = [web_search_tool]
            output_type = RevisedArticle # EditorはJSONを期待
        elif agent_type == "writer":
            base_prompt = SECTION_WRITER_AGENT_BASE_PROMPT
            instructions_func = create_section_writer_instructions
            tools = [web_search_tool]
            output_type = None # Writerは単純なテキスト出力を期待
        elif agent_type == "researcher":
            base_prompt = RESEARCHER_AGENT_BASE_PROMPT
            instructions_func = create_researcher_instructions
            tools = [web_search_tool]
            output_type = ResearchQueryResult # ResearcherはJSONを期待
            # LiteLLM経由でWebSearchToolを使う場合、tool_choiceがどう機能するか不明瞭
            # model_settings = ModelSettings(tool_choice={"type": "web_search"}) # これはOpenAI API特有の可能性
        else:
            console.print(f"[red]エラー: 未知のエージェントタイプ '{agent_type}'[/red]")
            return None

        litellm_agent = Agent[ArticleContext](
            name=agent_name,
            instructions=instructions_func(base_prompt),
            model=litellm_model_instance,
            model_settings=model_settings, # 必要に応じて設定
            tools=tools,
            output_type=output_type,
        )
        console.print(f"[green]LiteLLMモデル '{model_name}' を使用する {agent_type} エージェントを準備しました。[/green]")
        return litellm_agent
    except Exception as e:
        console.print(f"[bold red]LiteLLMモデル '{model_name}' ({agent_type}) の設定中にエラーが発生しました: {e}[/bold red]")
        return None

# --- ヘルパー関数 ---
# (変更なし)
def display_article_preview(html_content: str, title: str = "記事プレビュー"):
    """HTMLコンテンツをコンソールに簡易表示する"""
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    # HTMLタグを除去してテキストのみ表示（簡易的な方法）
    # より正確なプレビューにはライブラリ（例: beautifulsoup4）が必要
    preview_text = re.sub('<br>', '\n', html_content) # 改行を保持
    preview_text = re.sub('<[^>]+>', '', preview_text) # 他のタグを除去
    preview_text = preview_text.strip() # 前後の空白を除去

    max_preview_length = 1000
    if len(preview_text) > max_preview_length:
        preview_text = preview_text[:max_preview_length] + "..."
    console.print(preview_text)
    console.rule()

def save_article(html_content: str, title: Optional[str] = None, filename: str = "generated_article.html"):
    """生成されたHTMLを指定ファイル名で保存する"""
    try:
        filepath = Path(filename)
        # 記事全体を<html><body>タグで囲む（オプション）
        page_title = title or filepath.stem
        full_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }}
        h1, h2, h3 {{ margin-top: 1.5em; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 0.5em; }}
        strong {{ font-weight: bold; }}
        em {{ font-style: italic; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
<h1>{page_title}</h1>
{html_content}
</body>
</html>"""
        filepath.write_text(full_html, encoding="utf-8")
        console.print(f"[green]記事を {filepath.resolve()} に保存しました。[/green]")
    except Exception as e:
        console.print(f"[bold red]記事の保存中にエラーが発生しました: {e}[/bold red]")

# --- メイン実行ループ ---
async def run_main_loop(context: ArticleContext, run_config: RunConfig):
    """エージェントとの対話ループを実行する関数"""
    current_agent: Optional[Agent[ArticleContext]] = None
    # agent_input: Union[str, List[MessageInputItem]] # 修正: 型ヒントを変更
    agent_input: Union[str, List[Dict[str, Any]]] # 修正: Dict[str, Any]を使用

    while context.current_step not in ["completed", "error"]:
        console.rule(f"[bold yellow]現在のステップ: {context.current_step}[/bold yellow]")

        # --- ステップに応じたエージェントと入力の決定 ---
        if context.current_step == "start":
            current_agent = theme_agent
            agent_input = f"キーワード「{', '.join(context.initial_keywords)}」とペルソナ「{context.target_persona}」に基づいて、{context.num_theme_proposals}個のテーマ案を生成してください。"
            console.print(f"🤖 {current_agent.name} にテーマ提案を依頼します...")

        elif context.current_step == "theme_proposed":
            # ユーザーにテーマ選択を促す (変更なし)
            if context.last_agent_output and isinstance(context.last_agent_output, ThemeProposal):
                console.print("[bold cyan]提案されたテーマ:[/bold cyan]")
                for i, theme in enumerate(context.last_agent_output.themes):
                    console.print(f"  [bold]{i+1}. {theme.title}[/bold]")
                    console.print(f"      説明: {theme.description}")
                    console.print(f"      キーワード: {', '.join(theme.keywords)}")
                while True:
                    try:
                        choice = Prompt.ask(f"使用するテーマの番号を選択してください (1-{len(context.last_agent_output.themes)})", default="1")
                        selected_index = int(choice) - 1
                        if 0 <= selected_index < len(context.last_agent_output.themes):
                            context.selected_theme = context.last_agent_output.themes[selected_index]
                            context.current_step = "theme_selected" # 次のステップへ
                            console.print(f"[green]テーマ「{context.selected_theme.title}」が選択されました。[/green]")
                            break
                        else: console.print("[yellow]無効な番号です。[/yellow]")
                    except ValueError: console.print("[yellow]数値を入力してください。[/yellow]")
            else:
                context.error_message = "テーマ提案の取得に失敗しました。"
                context.current_step = "error"
            continue # ユーザー入力待ち

        elif context.current_step == "theme_selected":
            # 次はリサーチ計画ステップへ
            context.current_step = "research_planning"
            console.print("リサーチ計画ステップに進みます...")
            continue # エージェント実行なし

        # --- リサーチフェーズ ---
        elif context.current_step == "research_planning":
            current_agent = research_planner_agent
            agent_input = f"選択されたテーマ「{context.selected_theme.title if context.selected_theme else ''}」についてのリサーチ計画を作成してください。"
            console.print(f"🤖 {current_agent.name} にリサーチ計画作成を依頼します...")

        elif context.current_step == "research_plan_generated":
            # リサーチ計画確認 (オプション)
            if context.research_plan:
                console.print("[bold cyan]生成されたリサーチ計画:[/bold cyan]")
                console.print(f"トピック: {context.research_plan.topic}")
                for i, q in enumerate(context.research_plan.queries):
                    console.print(f"  クエリ {i+1}: {q.query} (焦点: {q.focus})")
                confirm = Prompt.ask("この計画でリサーチを開始しますか？ (y/n)", choices=["y", "n"], default="y")
                if confirm.lower() == 'y':
                    context.current_step = "researching"
                    context.current_research_query_index = 0 # 最初のクエリから開始
                    context.research_query_results = [] # 結果リストを初期化
                else:
                    console.print("[yellow]リサーチ計画を修正するか、前のステップに戻ってください。（現実装では終了します）[/yellow]")
                    context.current_step = "error"
                    context.error_message = "ユーザーがリサーチ計画を拒否しました。"
            else:
                context.error_message = "リサーチ計画の取得に失敗しました。"
                context.current_step = "error"
            continue # ユーザー確認

        elif context.current_step == "researching":
            if not context.research_plan or context.current_research_query_index >= len(context.research_plan.queries):
                # 全クエリのリサーチ完了 -> 要約ステップへ
                context.current_step = "research_synthesizing"
                console.print("[green]全クエリのリサーチが完了しました。要約ステップに移ります。[/green]")
                continue

            current_agent = researcher_agent
            current_query_obj = context.research_plan.queries[context.current_research_query_index]
            agent_input = f"リサーチ計画のクエリ {context.current_research_query_index + 1}「{current_query_obj.query}」について調査し、結果を詳細に抽出・要約してください。" # 修正: 指示変更
            console.print(f"🤖 {current_agent.name} にクエリ {context.current_research_query_index + 1}/{len(context.research_plan.queries)} の詳細リサーチを依頼します...")

        elif context.current_step == "research_synthesizing":
            current_agent = research_synthesizer_agent
            agent_input = "収集された詳細なリサーチ結果を分析し、記事執筆のための詳細な要約レポートを作成してください。" # 修正: 入力変更
            console.print(f"🤖 {current_agent.name} に詳細リサーチ結果の要約を依頼します...")

        elif context.current_step == "research_report_generated":
             # リサーチレポート確認 (オプション、詳細表示)
            if context.research_report:
                console.print("[bold cyan]生成された詳細リサーチレポート:[/bold cyan]")
                console.print(f"トピック: {context.research_report.topic}")
                console.print(f"全体要約: {context.research_report.overall_summary}")
                console.print("主要ポイントと出典:")
                for kp in context.research_report.key_points:
                    sources_str = ", ".join(kp.supporting_sources[:3]) # 代表的なソースをいくつか表示
                    if len(kp.supporting_sources) > 3: sources_str += ", ..."
                    console.print(f"  - {kp.point} (出典: {sources_str})")
                console.print("面白い切り口:")
                for a in context.research_report.interesting_angles: console.print(f"  - {a}")
                console.print(f"全情報源URL数: {len(context.research_report.all_sources)}")

                confirm = Prompt.ask("このレポートを基にアウトライン作成に進みますか？ (y/n)", choices=["y", "n"], default="y")
                if confirm.lower() == 'y':
                    context.current_step = "outline_generation" # アウトライン生成ステップへ
                else:
                    console.print("[yellow]リサーチをやり直すか、前のステップに戻ってください。（現実装では終了します）[/yellow]")
                    context.current_step = "error"
                    context.error_message = "ユーザーがリサーチレポートを拒否しました。"
            else:
                context.error_message = "リサーチレポートの取得に失敗しました。"
                context.current_step = "error"
            continue # ユーザー確認

        # --- アウトライン作成フェーズ ---
        elif context.current_step == "outline_generation": # 新しいステップ名
            current_agent = outline_agent
            agent_input = f"選択されたテーマ「{context.selected_theme.title if context.selected_theme else ''}」、詳細リサーチレポート、目標文字数 {context.target_length or '指定なし'} に基づいてアウトラインを作成してください。" # 修正: 入力変更
            console.print(f"🤖 {current_agent.name} にアウトライン作成を依頼します...")

        elif context.current_step == "outline_generated":
            # アウトライン確認 (変更なし、ただし次のステップは writing_sections)
            if context.generated_outline:
                console.print("[bold cyan]生成されたアウトライン:[/bold cyan]")
                console.print(f"タイトル: {context.generated_outline.title}")
                console.print(f"トーン: {context.generated_outline.suggested_tone}")
                for i, section in enumerate(context.generated_outline.sections):
                    console.print(f"  {i+1}. {section.heading}") # ここはユーザー向け表示なので1ベースでOK
                    # サブセクション表示は省略
                confirm = Prompt.ask("このアウトラインで記事生成を開始しますか？ (y/n)", choices=["y", "n"], default="y")
                if confirm.lower() == 'y':
                    context.current_step = "writing_sections"
                    context.current_section_index = 0 # 内部インデックスは0から
                    context.generated_sections_html = [] # HTMLリスト初期化
                    context.clear_section_writer_history() # ライター履歴初期化
                    # 最初のシステムプロンプトを履歴に追加 (修正: プロンプト関数呼び出しをawait)
                    base_instruction_text = await create_section_writer_instructions(SECTION_WRITER_AGENT_BASE_PROMPT)(RunContextWrapper(context=context), section_writer_agent)
                    # 修正: 'system' ロールを使用
                    context.add_to_section_writer_history("system", base_instruction_text)
                else:
                    console.print("[yellow]アウトラインを修正するか、前のステップに戻ってください。（現実装では終了します）[/yellow]")
                    context.current_step = "error" # 簡単のため終了
                    context.error_message = "ユーザーがアウトラインを拒否しました。"
            else:
                 context.error_message = "アウトラインの取得に失敗しました。"
                 context.current_step = "error"
            continue

        # --- 執筆フェーズ (修正: Streaming対応) ---
        elif context.current_step == "writing_sections":
            if not context.generated_outline or context.current_section_index >= len(context.generated_outline.sections):
                context.full_draft_html = context.get_full_draft()
                context.current_step = "editing"
                console.print("[green]全セクションの執筆が完了しました。編集ステップに移ります。[/green]")
                continue

            current_agent = section_writer_agent
            # 内部インデックス (0ベース) を使用
            target_index = context.current_section_index
            target_heading = context.generated_outline.sections[target_index].heading

            # --- 会話履歴を input として構築 ---
            # 履歴は context.section_writer_history に蓄積されている
            # 今回の執筆依頼を user メッセージとして追加
            # ユーザー向け表示は1ベース、内部処理・プロンプト指示は0ベース
            user_request = f"前のセクション（もしあれば）に続けて、アウトラインのセクション {target_index + 1}「{target_heading}」の内容をHTMLで執筆してください。提供された詳細リサーチ情報を参照し、必要に応じて出典へのリンクを含めてください。" # 修正: リンク生成を促す

            # 現在の履歴に今回のユーザーリクエストを追加して agent_input とする
            # 注意: context.section_writer_history 自体は変更せず、Runnerに渡すリストを作成
            # current_input_messages: List[MessageInputItem] = list(context.section_writer_history) # 修正: 型ヒントを変更
            current_input_messages: List[Dict[str, Any]] = list(context.section_writer_history) # 修正: Dict[str, Any]を使用
            current_input_messages.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user_request}]
            })
            agent_input = current_input_messages # Runnerにはリスト形式で渡す
            # ------------------------------------

            # ユーザー向け表示は1ベース、内部処理は0ベース
            console.print(f"🤖 {current_agent.name} にセクション {target_index + 1} の執筆を依頼します (Streaming)...")

            # --- Streaming 実行 ---
            stream_result = None
            last_exception = None
            accumulated_html = ""
            for attempt in range(MAX_RETRIES):
                try:
                    stream_result = Runner.run_streamed( # run_streamed を使用
                        starting_agent=current_agent,
                        input=agent_input, # 会話履歴リスト
                        context=context,
                        run_config=run_config,
                        max_turns=10 # ツール使用等を考慮したターン数
                    )

                    console.print(f"[dim]ストリーム開始: セクション {target_index + 1}「{target_heading}」[/dim]")
                    accumulated_html = "" # 各リトライでリセット
                    async for event in stream_result.stream_events():
                        # Raw response イベント (テキストデルタ) を処理
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            delta = event.data.delta
                            print(delta, end="", flush=True) # リアルタイムで出力
                            accumulated_html += delta
                        # 他のイベントタイプも必要に応じて処理 (例: tool_call, agent_updated)
                        elif event.type == "run_item_stream_event":
                            if event.item.type == "tool_call_item":
                                console.print(f"\n[dim]ツール呼び出し: {event.item.name}[/dim]")
                            elif event.item.type == "tool_call_output_item":
                                console.print(f"\n[dim]ツール出力受信[/dim]") # 出力内容は大きい可能性があるので省略
                        elif event.type == "agent_updated_stream_event":
                             console.print(f"\n[dim]エージェント更新: {event.new_agent.name}[/dim]")
                        # 完了イベントを検出 (オプション)
                        elif event.type == "raw_response_event" and isinstance(event.data, ResponseCompletedEvent):
                             console.print(f"\n[dim]レスポンス完了イベント受信[/dim]")

                    console.print(f"\n[dim]ストリーム終了: セクション {target_index + 1}「{target_heading}」[/dim]")

                    # ストリームが正常に完了したらリトライループを抜ける
                    last_exception = None
                    break

                except (InternalServerError, BadRequestError, MaxTurnsExceeded, ModelBehaviorError, AgentsException, UserError, Exception) as e:
                    last_exception = e
                    console.print(f"\n[yellow]ストリーミング中にエラー発生 (試行 {attempt + 1}/{MAX_RETRIES}): {type(e).__name__} - {e}[/yellow]")
                    if isinstance(e, InternalServerError) and attempt < MAX_RETRIES - 1:
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt) # Exponential backoff
                        await asyncio.sleep(delay)
                    else:
                        # リトライしないエラーまたは最終リトライ
                        context.error_message = f"ストリーミングエラー: {e}"
                        context.current_step = "error"
                        break # リトライループを抜ける

            # ストリーミング後の処理
            if context.current_step == "error":
                continue # エラーが発生したら次のループへ

            if last_exception: # リトライしてもダメだった場合
                 context.error_message = f"ストリーミングエラー（リトライ上限到達）: {last_exception}"
                 context.current_step = "error"
                 continue

            # --- ストリーム結果から ArticleSection を手動で構築 ---
            if accumulated_html:
                # 期待されるインデックスと見出しを使用して ArticleSection を作成
                try:
                    generated_section = ArticleSection(
                        section_index=target_index,
                        heading=target_heading,
                        html_content=accumulated_html.strip() # 前後の空白を除去
                    )
                    console.print(f"[green]セクション {target_index + 1}「{generated_section.heading}」のHTMLをストリームから構築しました。[/green]")
                    display_article_preview(generated_section.html_content, f"セクション {target_index + 1} プレビュー (Streamed)")

                    # コンテキストを更新
                    context.generated_sections_html.append(generated_section.html_content)
                    context.last_agent_output = generated_section # last_agent_output に構築したオブジェクトを格納

                    # 会話履歴を更新 (ユーザーリクエスト + 生成されたHTML)
                    last_user_request_item = agent_input[-1] if agent_input and isinstance(agent_input, list) else None
                    if last_user_request_item and last_user_request_item.get('role') == 'user':
                         # ユーザーリクエストのテキストを取得
                         user_request_text = last_user_request_item['content'][0]['text']
                         context.add_to_section_writer_history("user", user_request_text)
                    else:
                         # フォールバック
                         context.add_to_section_writer_history("user", f"セクション {target_index + 1} の執筆依頼（履歴復元失敗）")

                    context.add_to_section_writer_history("assistant", generated_section.html_content)

                    # 次のセクションへ
                    context.current_step = "writing_sections" # ステップは維持
                    context.current_section_index += 1

                except ValidationError as e:
                    console.print(f"[bold red]エラー: 構築した ArticleSection のバリデーションに失敗しました。[/bold red]")
                    console.print(f"エラー詳細: {e}")
                    console.print(f"Accumulated HTML: {accumulated_html[:500]}...")
                    context.error_message = f"ArticleSection構築エラー: {e}"
                    context.current_step = "error"

            else:
                console.print(f"[yellow]警告: セクション {target_index + 1} のHTMLコンテンツがストリームから取得できませんでした。[/yellow]")
                # 空のセクションとして扱うか、エラーにするか選択
                # ここではエラーにする
                context.error_message = f"セクション {target_index + 1} のHTMLコンテンツが空です。"
                context.current_step = "error"

            continue # writing_sections ステップの次のイテレーションへ

        # --- 編集フェーズ ---
        elif context.current_step == "editing":
            # LiteLLM選択ロジック (変更なし)
            use_litellm = Prompt.ask("編集にLiteLLMモデルを使用しますか？ (y/n)", choices=["y", "n"], default="n")
            if use_litellm.lower() == 'y' and LITELLM_AVAILABLE:
                litellm_model_name = Prompt.ask("[cyan]使用するLiteLLMモデル名を入力してください (例: litellm/anthropic/claude-3-haiku-20240307)[/cyan]", default="litellm/anthropic/claude-3-haiku-20240307")
                # APIキーは環境変数から取得するか、ここでプロンプト表示など
                litellm_api_key = os.getenv(f"{litellm_model_name.split('/')[1].upper()}_API_KEY") # 例: ANTHROPIC_API_KEY
                if not litellm_api_key:
                    console.print(f"[yellow]警告: {litellm_model_name} のAPIキーが環境変数等で見つかりません。デフォルトのOpenAIモデルを使用します。[/yellow]")
                    current_agent = editor_agent
                else:
                    litellm_editor = get_litellm_agent("editor", litellm_model_name, litellm_api_key)
                    current_agent = litellm_editor if litellm_editor else editor_agent
            else:
                 current_agent = editor_agent

            agent_input = "記事ドラフト全体をレビューし、詳細リサーチ情報に基づいて推敲・編集してください。特にリンクの適切性を確認してください。" # 修正: 指示変更
            console.print(f"🤖 {current_agent.name} に最終編集を依頼します...")

        else:
            console.print(f"[bold red]未定義のステップ: {context.current_step}[/bold red]")
            context.current_step = "error"
            context.error_message = f"未定義のステップ {context.current_step} に到達しました。"
            continue

        # --- Agent実行 (リトライロジック付き、SectionWriter以外) ---
        if context.current_step != "writing_sections": # SectionWriterは上でStreaming処理済み
            if not current_agent:
                 context.error_message = "実行するエージェントが見つかりません。"
                 context.current_step = "error"
                 continue

            result = None
            last_exception = None
            for attempt in range(MAX_RETRIES):
                try:
                    console.print(f"[dim]エージェント {current_agent.name} 実行開始 (試行 {attempt + 1}/{MAX_RETRIES})...[/dim]")
                    result = await Runner.run( # 通常の run を使用
                        starting_agent=current_agent,
                        input=agent_input, # 文字列または会話履歴リスト
                        context=context,
                        run_config=run_config,
                        max_turns=10 # ツール使用等を考慮したターン数
                    )
                    console.print(f"[dim]エージェント {current_agent.name} 実行完了。[/dim]")
                    # 成功したらループを抜ける
                    last_exception = None
                    break
                except InternalServerError as e:
                    last_exception = e
                    console.print(f"[yellow]サーバーエラー (5xx) が発生しました。リトライします... ({attempt + 1}/{MAX_RETRIES}) 詳細: {e}[/yellow]")
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt) # Exponential backoff
                    await asyncio.sleep(delay)
                except BadRequestError as e:
                    # BadRequestError (4xx) はリトライしないことが多い
                    last_exception = e
                    console.print(f"[bold red]リクエストエラー (4xx) が発生しました: {e}[/bold red]")
                    # エラーメッセージにリクエスト内容の一部を含めるとデバッグに役立つ場合がある
                    # input_str = str(agent_input)[:500] # 長すぎる場合は切り詰める
                    # console.print(f"Request Input (partial): {input_str}")
                    context.error_message = f"リクエストエラー: {e}"
                    context.current_step = "error"
                    break
                except (MaxTurnsExceeded, ModelBehaviorError, AgentsException, UserError) as e:
                    # これらはリトライ対象外のエラー
                    last_exception = e
                    console.print(f"[bold red]Agent実行エラー ({type(e).__name__}): {e}[/bold red]")
                    context.error_message = f"Agent実行エラー: {e}"
                    context.current_step = "error"
                    break # リトライせずにループを抜ける
                except Exception as e:
                    # その他の予期せぬエラー
                    last_exception = e
                    console.print(f"[bold red]予期せぬエラーが発生しました: {e}[/bold red]")
                    console.print(traceback.format_exc()) # スタックトレースを出力
                    context.error_message = f"予期せぬエラー: {e}"
                    context.current_step = "error"
                    break # リトライせずにループを抜ける

            # リトライしてもエラーが解消しなかった場合
            if last_exception:
                if not context.error_message: # エラーメッセージがまだ設定されていなければ設定
                       context.error_message = f"Agent実行中にエラーが発生しました（リトライ上限到達）: {last_exception}"
                context.current_step = "error"
                continue # エラーステップへ

            # --- 結果処理 (SectionWriter以外) ---
            agent_output: Optional[AgentOutput] = None
            raw_output_text = "" # デバッグ用
            if result and result.final_output: # resultがNoneでないことを確認
                 raw_output_text = str(result.final_output) # デバッグ用に保持
                 if isinstance(result.final_output, AgentOutput.__args__): # type: ignore
                      agent_output = result.final_output
                 elif isinstance(result.final_output, str):
                      try:
                          # JSON文字列としてパースを試みる
                          parsed_output = json.loads(result.final_output)
                          # 型アノテーションを使って適切なPydanticモデルに変換
                          status = parsed_output.get("status")
                          output_model : Optional[type[BaseModel]] = None
                          if status == "theme_proposal": output_model = ThemeProposal
                          elif status == "outline": output_model = Outline
                          # elif status == "article_section": output_model = ArticleSection # SectionWriterは別処理
                          elif status == "revised_article": output_model = RevisedArticle
                          elif status == "clarification_needed": output_model = ClarificationNeeded
                          elif status == "status_update": output_model = StatusUpdate
                          elif status == "research_plan": output_model = ResearchPlan
                          elif status == "research_query_result": output_model = ResearchQueryResult # 修正: 型追加
                          elif status == "research_report": output_model = ResearchReport # 修正: 型追加

                          if output_model:
                               agent_output = output_model.model_validate(parsed_output)
                          else:
                               raise ValueError(f"未知のstatus: {status}")

                      except (json.JSONDecodeError, ValidationError, ValueError) as parse_error:
                          console.print(f"[yellow]警告: Agentからの応答が予期したJSON形式ではありません。内容: {result.final_output[:100]}... エラー: {parse_error}[/yellow]")
                          agent_output = StatusUpdate(status="status_update", message=f"エージェントからの非構造応答: {result.final_output[:100]}...")
                 # Pydanticモデルでもなく、JSON文字列でもない場合
                 else:
                      console.print(f"[yellow]警告: Agentからの応答が予期した型(Pydantic/JSON str)ではありません。型: {type(result.final_output)}, 内容: {str(result.final_output)[:100]}...[/yellow]")
                      agent_output = StatusUpdate(status="status_update", message=f"エージェントからの予期せぬ型応答: {str(result.final_output)[:100]}...")

            context.last_agent_output = agent_output # SectionWriter以外は AgentOutput 型

            if not agent_output:
                 console.print(f"[yellow]エージェントから有効な出力が得られませんでした。Raw Output: {raw_output_text[:200]}[/yellow]")
                 context.error_message = "エージェントから有効な出力が得られませんでした。"
                 context.current_step = "error"
                 continue

            # --- ステップ更新 (SectionWriter以外) ---
            if isinstance(agent_output, ThemeProposal):
                context.current_step = "theme_proposed"
            elif isinstance(agent_output, ResearchPlan):
                context.research_plan = agent_output
                context.current_step = "research_plan_generated"
            elif isinstance(agent_output, ResearchQueryResult) and context.current_step == "researching": # 修正: 型チェック変更
                if context.research_plan and agent_output.query == context.research_plan.queries[context.current_research_query_index].query:
                    context.add_query_result(agent_output) # 修正: メソッド呼び出し
                    console.print(f"[green]クエリ「{agent_output.query}」の詳細リサーチ結果を処理しました。[/green]")
                    context.current_research_query_index += 1 # 次のクエリへ
                else:
                     console.print(f"[yellow]警告: 予期しないクエリ「{agent_output.query}」の結果を受け取りました。[/yellow]")
                     context.error_message = "予期しないクエリの結果。"
                     context.current_step = "error"
                # researching ステップは継続
            elif isinstance(agent_output, ResearchReport): # 修正: 型チェック変更
                context.research_report = agent_output
                context.current_step = "research_report_generated"
            elif isinstance(agent_output, Outline):
                context.generated_outline = agent_output
                context.current_step = "outline_generated"
            # ArticleSection の処理は writing_sections ステップ内で行う
            elif isinstance(agent_output, RevisedArticle):
                context.final_article_html = agent_output.final_html_content
                context.current_step = "completed"
                console.print("[green]記事の編集が完了しました！[/green]")
                display_article_preview(context.final_article_html, "完成記事プレビュー")
            elif isinstance(agent_output, ClarificationNeeded):
                console.print(f"[bold yellow]確認が必要です:[/bold yellow] {agent_output.message}")
                context.error_message = f"ユーザーへの確認が必要: {agent_output.message}"
                context.current_step = "error"
            elif isinstance(agent_output, StatusUpdate):
                 console.print(f"[cyan]ステータス:[/cyan] {agent_output.message}")
                 # 特に何もしない

    # --- ループ終了後 ---
    if context.current_step == "completed":
        console.print("\n🎉 [bold green]SEO記事の生成が正常に完了しました。[/bold green]")
        if context.final_article_html:
             save_confirm = Prompt.ask("最終記事をHTMLファイルとして保存しますか？ (y/n)", choices=["y", "n"], default="y")
             if save_confirm.lower() == 'y':
                 # タイトルをOutlineから取得してファイル名にする
                 filename = "final_article.html"
                 article_title = "生成された記事"
                 if context.generated_outline and context.generated_outline.title:
                     article_title = context.generated_outline.title
                     # ファイル名に使えない文字を除去・置換
                     safe_title = re.sub(r'[\\/*?:"<>|]', '_', article_title)
                     filename = f"{safe_title}.html"
                 save_article(context.final_article_html, title=article_title, filename=filename)
        else:
             console.print("[yellow]警告: 最終記事コンテンツが見つかりません。[/yellow]")

    elif context.current_step == "error":
        console.print(f"\n❌ [bold red]記事生成プロセス中にエラーが発生しました。[/bold red]")
        if context.error_message:
            console.print(f"エラー詳細: {context.error_message}")

    console.print("プログラムを終了します。")


async def main():
    console.print("[bold magenta]📝 SEO記事生成システム (リサーチ強化・リンク生成版) へようこそ！[/bold magenta]") # タイトル変更

    # --- ユーザーからの初期情報入力 ---
    keywords_str = Prompt.ask("[cyan]ターゲットキーワードを入力してください（カンマ区切り）[/cyan]", default="札幌, 注文住宅, 自然素材, 子育て") # 例を更新
    initial_keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

    target_persona = Prompt.ask("[cyan]ターゲットペルソナを入力してください（例：庭づくり初心者, 子供がいる家庭）[/cyan]", default="札幌近郊で自然素材を使った家づくりに関心がある、小さな子供を持つ30代夫婦") # 例を更新

    target_length_str = Prompt.ask("[cyan]目標文字数を入力してください（任意、数値のみ）[/cyan]", default="3000") # 例を更新
    target_length = None
    if target_length_str.isdigit():
        target_length = int(target_length_str)

    num_themes_str = Prompt.ask("[cyan]提案してほしいテーマ数を入力してください[/cyan]", default="3")
    num_theme_proposals = 3
    if num_themes_str.isdigit() and int(num_themes_str) > 0:
        num_theme_proposals = int(num_themes_str)

    num_research_queries_str = Prompt.ask("[cyan]リサーチで使用する検索クエリ数を入力してください[/cyan]", default="5") # 例を更新
    num_research_queries = 5
    if num_research_queries_str.isdigit() and int(num_research_queries_str) > 0:
        num_research_queries = int(num_research_queries_str)

    vector_store_id = Prompt.ask("[cyan]File Searchで使用するVector Store IDを入力してください（任意）[/cyan]", default="")


    # --- コンテキスト初期化 ---
    article_context = ArticleContext(
        initial_keywords=initial_keywords,
        target_persona=target_persona,
        target_length=target_length,
        num_theme_proposals=num_theme_proposals,
        num_research_queries=num_research_queries,
        vector_store_id=vector_store_id if vector_store_id else None,
        # ダミーの会社情報を設定（オプション）
        company_name="株式会社ナチュラルホームズ札幌", # 例を更新
        company_description="札幌を拠点に、自然素材を活かした健康で快適な注文住宅を提供しています。", # 例を更新
        company_style_guide="専門用語を避け、温かみのある丁寧語（ですます調）で。子育て世代の読者に寄り添い、安心感を与えるようなトーンを心がける。", # 例を更新
    )

    # --- 実行設定 ---
    run_config = RunConfig(
        workflow_name="SEOArticleGenerationDeepResearchJP", # ワークフロー名を変更
        trace_id=f"trace_{uuid.uuid4().hex}",
        # trace_include_sensitive_data=False # 必要に応じて機密データトレースを無効化
    )

    # --- メインループ実行 ---
    await run_main_loop(article_context, run_config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n[bold red]プログラム実行中に致命的なエラーが発生しました: {e}[/bold red]")
        console.print(traceback.format_exc()) # スタックトレースを出力

