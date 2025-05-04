# -*- coding: utf-8 -*-
import os
import json
import asyncio
import re
import time # リトライのためのtimeモジュールをインポート
from pathlib import Path
from openai import AsyncOpenAI, BadRequestError, InternalServerError # InternalServerErrorをインポート
from dotenv import load_dotenv
import rich.console
import rich.prompt
import rich.syntax
from typing import List, Dict, Union, Optional, Tuple, Any, Literal, Callable, Awaitable
from pydantic import BaseModel, Field, ValidationError, field_validator
from dataclasses import dataclass, field
import uuid

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
console = rich.console.Console()
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
DEFAULT_MODEL = "gpt-4o-mini"
RESEARCH_MODEL = "gpt-4o-mini" # リサーチもminiで試す
WRITING_MODEL = "o4-mini"  # 執筆もminiで試す
EDITING_MODEL = "gpt-4o-mini"  # 編集もminiで試す

# リトライ設定
MAX_RETRIES = 3 # 最大リトライ回数
INITIAL_RETRY_DELAY = 1 # 初期リトライ遅延（秒）

# --- Pydanticモデル定義 (Agentの出力型) ---
# (変更なし)
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

class ArticleSection(BaseModel):
    """生成された記事の単一セクション"""
    status: Literal["article_section"] = Field(description="出力タイプ: 記事セクション")
    section_index: int = Field(description="生成対象のセクションインデックス（Outline.sectionsのインデックス）")
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

# --- リサーチ関連モデル ---
class ResearchQuery(BaseModel):
    """リサーチプラン内の単一検索クエリ"""
    query: str = Field(description="実行する具体的な検索クエリ")
    focus: str = Field(description="このクエリで特に調査したい点")

class ResearchPlan(BaseModel):
    """リサーチ計画"""
    status: Literal["research_plan"] = Field(description="出力タイプ: リサーチ計画")
    topic: str = Field(description="リサーチ対象のトピック（記事テーマ）")
    queries: List[ResearchQuery] = Field(description="実行する検索クエリのリスト")

class ResearchQueryResult(BaseModel): # 新しいモデル
    """単一クエリのリサーチ結果要約"""
    status: Literal["research_query_result"] = Field(description="出力タイプ: リサーチクエリ結果")
    query: str = Field(description="実行された検索クエリ")
    summary: str = Field(description="検索結果の主要な情報の要約")
    relevant_snippets: List[str] = Field(description="記事作成に役立ちそうな短い抜粋")
    source_urls: List[str] = Field(description="参照した主要な情報源URL")

class ResearchReport(BaseModel):
    """リサーチ結果の要約レポート"""
    status: Literal["research_report"] = Field(description="出力タイプ: リサーチレポート")
    topic: str = Field(description="リサーチ対象のトピック")
    overall_summary: str = Field(description="リサーチ全体から得られた主要な洞察やポイントの要約")
    key_points: List[str] = Field(description="記事に含めるべき重要なポイントや事実のリスト")
    interesting_angles: List[str] = Field(description="記事を面白くするための切り口や視点のアイデア")
    sources_used: List[str] = Field(description="参照した主要な情報源URLのリスト")

# エージェントが出力しうる型のUnion
AgentOutput = Union[
    ThemeProposal, Outline, ArticleSection, RevisedArticle, ClarificationNeeded, StatusUpdate,
    ResearchPlan, ResearchQueryResult, ResearchReport # ResearchSnippetを削除し、ResearchQueryResultを追加
]

# --- コンテキストクラス ---
@dataclass
class ArticleContext:
    """記事生成プロセス全体で共有されるコンテキスト"""
    # (変更なし)
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
    research_query_results: List[ResearchQueryResult] = field(default_factory=list) # 追加: クエリ結果を保存
    research_report: Optional[ResearchReport] = None # 最終リサーチレポート
    generated_outline: Optional[Outline] = None
    current_section_index: int = 0
    generated_sections_html: List[str] = field(default_factory=list) # 各セクションのHTMLを格納
    full_draft_html: Optional[str] = None # 結合後のドラフト
    final_article_html: Optional[str] = None # 最終成果物
    error_message: Optional[str] = None
    last_agent_output: Optional[AgentOutput] = None # 直前のエージェント出力を保持
    section_writer_history: List[Dict[str, Any]] = field(default_factory=list)

    def get_full_draft(self) -> str:
        """生成されたセクションを結合して完全なドラフトHTMLを返す"""
        return "\n".join(self.generated_sections_html)

    def add_query_result(self, result: ResearchQueryResult): # 新しいメソッド
        """リサーチクエリ結果を追加"""
        self.research_query_results.append(result)

    def clear_section_writer_history(self):
        """セクションライターの履歴をクリア"""
        self.section_writer_history = []


# --- ツール定義 ---
# (変更なし)
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
    return {
        "success": True,
        "summary": f"'{query}' に関する競合記事",
        "common_sections": ["ダミー"],
        "estimated_length_range": "1500〜3000文字",
    }

# --- 動的プロンプト生成関数 ---
# (変更なし)
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
- 検索結果を分析し、記事テーマとクエリの焦点に関連する**主要な情報を要約**してください。
- 特に記事作成に役立ちそうな**短い抜粋 (relevant_snippets)** をいくつか含めてください。
- 参照した**主要な情報源のURL (source_urls)** もリストアップしてください。
- あなたの応答は必ず `ResearchQueryResult` 型のJSON形式で出力してください。他のテキストは含めないでください。
- **`save_research_snippet` ツールは使用しないでください。**
"""
        return full_prompt
    return dynamic_instructions_func

def create_research_synthesizer_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.research_query_results:
            return "エラー: 要約するためのリサーチ結果がありません。"

        results_str = ""
        all_sources = set()
        for i, result in enumerate(ctx.context.research_query_results):
            results_str += f"--- クエリ結果 {i+1} ({result.query}) ---\n"
            results_str += f"要約: {result.summary}\n"
            results_str += "抜粋:\n"
            for snip in result.relevant_snippets:
                results_str += f"- {snip}\n"
            results_str += "情報源:\n"
            for url in result.source_urls:
                results_str += f"- {url}\n"
                all_sources.add(url)
            results_str += "\n"

        full_prompt = f"""{base_prompt}

--- リサーチ対象テーマ ---
{ctx.context.selected_theme.title if ctx.context.selected_theme else 'N/A'}

--- 収集されたリサーチ結果 ---
{results_str[:15000]}
{ "... (以下省略)" if len(results_str) > 15000 else "" }
---

**重要:**
- 上記のリサーチ結果全体を分析し、記事執筆に役立つように情報を統合・要約してください。
- 以下の要素を含む**カジュアルで実用的なレポート**を作成してください:
    - `overall_summary`: リサーチ全体から得られた主要な洞察やポイントの要約。
    - `key_points`: 記事に含めるべき重要なポイントや事実のリスト形式。
    - `interesting_angles`: 記事を面白くするための切り口や視点のアイデアのリスト形式。
    - `sources_used`: 参照した主要な情報源URLのリスト（重複は削除）。
- レポートは論文調ではなく、記事作成者がすぐに使えるような分かりやすい言葉で記述してください。
- あなたの応答は必ず `ResearchReport` 型のJSON形式で出力してください。
"""
        return full_prompt
    return dynamic_instructions_func

def create_outline_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.selected_theme or not ctx.context.research_report:
            return "エラー: テーマまたはリサーチレポートが利用できません。"

        company_info_str = f"文体ガイド: {ctx.context.company_style_guide}" if ctx.context.company_style_guide else "企業文体ガイドなし"
        research_summary = f"リサーチ要約: {ctx.context.research_report.overall_summary}\n主要ポイント: {', '.join(ctx.context.research_report.key_points)}\n面白い切り口: {', '.join(ctx.context.research_report.interesting_angles)}"

        full_prompt = f"""{base_prompt}

--- 入力情報 ---
選択されたテーマ:
  タイトル: {ctx.context.selected_theme.title}
  説明: {ctx.context.selected_theme.description}
  キーワード: {', '.join(ctx.context.selected_theme.keywords)}
ターゲット文字数: {ctx.context.target_length or '指定なし（標準的な長さで）'}
{company_info_str}
--- リサーチ結果 ---
{research_summary}
---

**重要:**
- 上記のテーマと**リサーチ結果**、そして競合分析の結果（ツール使用）に基づいて、記事のアウトラインを作成してください。
- リサーチ結果の主要ポイントや面白い切り口をアウトラインに反映させてください。日本のよくあるブログやコラムのように親しみやすいトーンでアウトラインを作成してください。
- あなたの応答は必ず `Outline` または `ClarificationNeeded` 型のJSON形式で出力してください。
- 文字数指定がある場合は、それに応じてセクション数や深さを調整してください。
"""
        return full_prompt
    return dynamic_instructions_func

def create_section_writer_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.generated_outline or ctx.context.current_section_index >= len(ctx.context.generated_outline.sections):
            return "エラー: 有効なアウトラインまたはセクションインデックスがありません。"

        target_section = ctx.context.generated_outline.sections[ctx.context.current_section_index]
        section_target_chars = None
        if ctx.context.target_length and len(ctx.context.generated_outline.sections) > 0:
            estimated_total_body_chars = ctx.context.target_length * 0.8
            section_target_chars = int(estimated_total_body_chars / len(ctx.context.generated_outline.sections))

        outline_context = "\n".join([f"- {s.heading}" for s in ctx.context.generated_outline.sections])
        research_context_summary = f"関連リサーチ要約: {ctx.context.research_report.overall_summary[:500]}..." if ctx.context.research_report else "リサーチ情報なし"

        full_prompt = f"""{base_prompt}

--- 入力情報 ---
記事タイトル: {ctx.context.generated_outline.title}
記事全体のキーワード: {', '.join(ctx.context.selected_theme.keywords) if ctx.context.selected_theme else 'N/A'}
記事全体のトーン: {ctx.context.generated_outline.suggested_tone}
記事のアウトライン（全体像）:
{outline_context}
{research_context_summary}

--- 今回の執筆対象セクション ---
セクションインデックス: {ctx.context.current_section_index}
見出し: {target_section.heading}
このセクションの目標文字数: {section_target_chars or '指定なし（適切に）'}
---

**重要:**
- あなたのタスクは、上記の「今回の執筆対象セクション」の内容をHTML形式で生成すること**だけ**です。
- **会話履歴（inputとして渡される直前のセクションの内容など）を考慮し、自然な流れで文章を続けてください。**
- 他のセクションの内容は生成しないでください。
- 必ず `<p>`, `<h2>`, `<h3>`, `<ul>`, `<li>`, `<strong>` などの基本的なHTMLタグを使用し、構造化されたコンテンツを生成してください。
- SEOを意識し、関連キーワードを自然に含めてください。
- 創造性を発揮し、読者にとって価値のあるオリジナルな文章を作成してください。
- あなたの応答は必ず `ArticleSection` 型のJSON形式で、`html_content` に生成したHTML文字列を入れて出力してください。HTML以外のテキスト（例：「承知しました」「以下に生成します」など）は絶対に含めないでください。
"""
        return full_prompt
    return dynamic_instructions_func

def create_editor_instructions(base_prompt: str) -> Callable[[RunContextWrapper[ArticleContext], Agent[ArticleContext]], Awaitable[str]]:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleContext], agent: Agent[ArticleContext]) -> str:
        if not ctx.context.full_draft_html:
            return "エラー: 編集対象のドラフト記事がありません。"

        research_summary = f"リサーチ要約: {ctx.context.research_report.overall_summary[:500]}..." if ctx.context.research_report else "リサーチ情報なし"

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
{research_summary}
---

**重要:**
- 上記のドラフトHTMLをレビューし、記事の要件と**リサーチ結果**に基づいて推敲・編集してください。
- チェックポイント:
    - 全体の流れと一貫性
    - 各セクションの内容の質と正確性 (リサーチ結果との整合性も)
    - 文法、スペル、誤字脱字
    - 指示されたトーンとスタイルガイドの遵守
    - ターゲットペルソナへの適合性
    - SEO最適化（キーワードの自然な使用、見出し構造）
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
    output_type=AgentOutput,
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

# 3. リサーチャーエージェント (修正: tool_choice追加)
RESEARCHER_AGENT_BASE_PROMPT = """
あなたは熟練のリサーチャーです。
指定された検索クエリでWeb検索を実行し、結果の中から記事テーマに関連する最も重要で信頼できる情報を要約し、指定された形式で返します。
**必ず web_search ツールを使用してください。**
"""
researcher_agent = Agent[ArticleContext](
    name="ResearcherAgent",
    instructions=create_researcher_instructions(RESEARCHER_AGENT_BASE_PROMPT),
    model=RESEARCH_MODEL,
    tools=[web_search_tool], # save_research_snippet を削除済み
    output_type=ResearchQueryResult, # ResearchQueryResult を返すように変更済み
)

# 4. リサーチシンセサイザーエージェント
RESEARCH_SYNTHESIZER_AGENT_BASE_PROMPT = """
あなたは情報を整理し、要点を抽出する専門家です。
収集された多数のリサーチ結果（要約と抜粋）を分析し、記事のテーマに沿って統合・要約し、記事作成者が活用しやすい実用的なリサーチレポートを作成します。
"""
research_synthesizer_agent = Agent[ArticleContext](
    name="ResearchSynthesizerAgent",
    instructions=create_research_synthesizer_instructions(RESEARCH_SYNTHESIZER_AGENT_BASE_PROMPT),
    model=RESEARCH_MODEL,
    tools=[], # 基本的にツールは不要
    output_type=AgentOutput, # ResearchReport
)

# --- 記事作成エージェント群 ---
# 5. アウトライン作成エージェント
OUTLINE_AGENT_BASE_PROMPT = """
あなたはSEO記事のアウトライン（構成案）を作成する専門家です。
選択されたテーマ、目標文字数、企業のスタイルガイド、そして**リサーチレポート**に基づいて、論理的で網羅的、かつ読者の興味を引く記事のアウトラインを生成します。
`analyze_competitors` ツールで競合記事の構成を調査し、差別化できる構成を考案します。
`get_company_data` ツールでスタイルガイドを確認します。
文字数指定に応じて、見出しの数や階層構造を適切に調整します。
記事全体のトーンも提案してください。
"""
outline_agent = Agent[ArticleContext](
    name="OutlineAgent",
    instructions=create_outline_instructions(OUTLINE_AGENT_BASE_PROMPT),
    model=WRITING_MODEL,
    tools=[analyze_competitors, get_company_data],
    output_type=AgentOutput, # Outline or ClarificationNeeded
)

# 6. セクション執筆エージェント
SECTION_WRITER_AGENT_BASE_PROMPT = """
あなたは指定された記事のセクション（見出し）に関する内容を執筆するプロのライターです。
記事全体のテーマ、アウトライン、キーワード、トーン、**会話履歴（前のセクション）**、そしてリサーチ結果に基づき、割り当てられた特定のセクションの内容を、創造的かつSEOを意識してHTML形式で執筆します。
必要に応じて `web_search` ツールで最新情報や詳細情報を調査し、内容を充実させます。
**あなたのタスクは、指示された1つのセクションのHTMLコンテンツを生成することだけです。** 読者を引きつけ、価値を提供するオリジナルな文章を作成してください。
"""
section_writer_agent = Agent[ArticleContext](
    name="SectionWriterAgent",
    instructions=create_section_writer_instructions(SECTION_WRITER_AGENT_BASE_PROMPT),
    model=WRITING_MODEL,
    tools=[web_search_tool],
    output_type=AgentOutput, # ArticleSection
)

# 7. 推敲・編集エージェント
EDITOR_AGENT_BASE_PROMPT = """
あなたはプロの編集者兼SEOスペシャリストです。
与えられた記事ドラフト（HTML形式）を、記事の要件（テーマ、キーワード、ペルソナ、文字数、トーン、スタイルガイド）と**リサーチ結果**を照らし合わせながら、徹底的にレビューし、推敲・編集します。
文章の流れ、一貫性、正確性、文法、読みやすさ、独創性、そしてSEO最適化の観点から、最高品質の記事に仕上げることを目指します。
必要であれば `web_search` ツールでファクトチェックや追加情報を調査します。
最終的な成果物として、編集済みの完全なHTMLコンテンツを出力します。
"""
editor_agent = Agent[ArticleContext](
    name="EditorAgent",
    instructions=create_editor_instructions(EDITOR_AGENT_BASE_PROMPT),
    model=EDITING_MODEL,
    tools=[web_search_tool],
    output_type=AgentOutput, # RevisedArticle
)

# --- LiteLLM 設定例 ---
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
        output_type = AgentOutput
        model_settings = None # LiteLLM用の設定は別途必要か確認

        if agent_type == "editor":
            base_prompt = EDITOR_AGENT_BASE_PROMPT
            instructions_func = create_editor_instructions
            tools = [web_search_tool]
        elif agent_type == "writer":
            base_prompt = SECTION_WRITER_AGENT_BASE_PROMPT
            instructions_func = create_section_writer_instructions
            tools = [web_search_tool]
        elif agent_type == "researcher":
            base_prompt = RESEARCHER_AGENT_BASE_PROMPT
            instructions_func = create_researcher_instructions
            tools = [web_search_tool]
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
    preview_text = re.sub('<[^<]+?>', '', html_content)
    max_preview_length = 1000
    if len(preview_text) > max_preview_length:
        preview_text = preview_text[:max_preview_length] + "..."
    console.print(preview_text)
    console.rule()

def save_article(html_content: str, filename: str = "generated_article.html"):
    """生成されたHTMLを指定ファイル名で保存する"""
    try:
        filepath = Path(filename)
        filepath.write_text(html_content, encoding="utf-8")
        console.print(f"[green]記事を {filepath.resolve()} に保存しました。[/green]")
    except Exception as e:
        console.print(f"[bold red]記事の保存中にエラーが発生しました: {e}[/bold red]")

# --- メイン実行ループ ---
async def run_main_loop(context: ArticleContext, run_config: RunConfig):
    """エージェントとの対話ループを実行する関数"""
    # (ループ内のロジックは変更なし)
    current_agent: Optional[Agent[ArticleContext]] = None
    agent_input: Union[str, List[Dict[str, Any]]] # Agentへの入力 (文字列 or 会話履歴リスト)

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
                    console.print(f"     説明: {theme.description}")
                    console.print(f"     キーワード: {', '.join(theme.keywords)}")
                while True:
                    try:
                        choice = rich.prompt.Prompt.ask(f"使用するテーマの番号を選択してください (1-{len(context.last_agent_output.themes)})", default="1")
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
                confirm = rich.prompt.Prompt.ask("この計画でリサーチを開始しますか？ (y/n)", choices=["y", "n"], default="y")
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
            agent_input = f"リサーチ計画のクエリ {context.current_research_query_index + 1}「{current_query_obj.query}」について調査し、結果を要約してください。" # 指示変更
            console.print(f"🤖 {current_agent.name} にクエリ {context.current_research_query_index + 1}/{len(context.research_plan.queries)} のリサーチを依頼します...")

        elif context.current_step == "research_synthesizing":
            current_agent = research_synthesizer_agent
            agent_input = "収集されたリサーチ結果を分析し、記事執筆のための要約レポートを作成してください。" # 入力変更
            console.print(f"🤖 {current_agent.name} にリサーチ結果の要約を依頼します...")

        elif context.current_step == "research_report_generated":
             # リサーチレポート確認 (オプション)
            if context.research_report:
                console.print("[bold cyan]生成されたリサーチレポート:[/bold cyan]")
                console.print(f"トピック: {context.research_report.topic}")
                console.print(f"要約: {context.research_report.overall_summary}")
                console.print("主要ポイント:")
                for p in context.research_report.key_points: console.print(f"  - {p}")
                console.print("面白い切り口:")
                for a in context.research_report.interesting_angles: console.print(f"  - {a}")
                console.print(f"情報源URL数: {len(context.research_report.sources_used)}")

                confirm = rich.prompt.Prompt.ask("このレポートを基にアウトライン作成に進みますか？ (y/n)", choices=["y", "n"], default="y")
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
            agent_input = f"選択されたテーマ「{context.selected_theme.title if context.selected_theme else ''}」、リサーチレポート、目標文字数 {context.target_length or '指定なし'} に基づいてアウトラインを作成してください。"
            console.print(f"🤖 {current_agent.name} にアウトライン作成を依頼します...")

        elif context.current_step == "outline_generated":
            # アウトライン確認 (変更なし、ただし次のステップは writing_sections)
            if context.generated_outline:
                console.print("[bold cyan]生成されたアウトライン:[/bold cyan]")
                console.print(f"タイトル: {context.generated_outline.title}")
                console.print(f"トーン: {context.generated_outline.suggested_tone}")
                for i, section in enumerate(context.generated_outline.sections):
                     console.print(f"  {i+1}. {section.heading}")
                     # サブセクション表示は省略
                confirm = rich.prompt.Prompt.ask("このアウトラインで記事生成を開始しますか？ (y/n)", choices=["y", "n"], default="y")
                if confirm.lower() == 'y':
                    context.current_step = "writing_sections"
                    context.current_section_index = 0
                    context.generated_sections_html = [] # HTMLリスト初期化
                    context.clear_section_writer_history() # ライター履歴初期化
                else:
                    console.print("[yellow]アウトラインを修正するか、前のステップに戻ってください。（現実装では終了します）[/yellow]")
                    context.current_step = "error" # 簡単のため終了
                    context.error_message = "ユーザーがアウトラインを拒否しました。"
            else:
                 context.error_message = "アウトラインの取得に失敗しました。"
                 context.current_step = "error"
            continue

        # --- 執筆フェーズ ---
        elif context.current_step == "writing_sections":
            if not context.generated_outline or context.current_section_index >= len(context.generated_outline.sections):
                context.full_draft_html = context.get_full_draft()
                context.current_step = "editing"
                console.print("[green]全セクションの執筆が完了しました。編集ステップに移ります。[/green]")
                continue

            current_agent = section_writer_agent
            target_heading = context.generated_outline.sections[context.current_section_index].heading

            # --- 会話履歴を input として構築 ---
            # 1. 基本的な指示 (developerロールが良いかもしれない)
            base_instruction = await create_section_writer_instructions(SECTION_WRITER_AGENT_BASE_PROMPT)(RunContextWrapper(context=context), current_agent) # ダミーのWrapperとAgentを渡す

            # MessageInputItem の代わりに辞書を使用
            current_input_messages: List[Dict[str, Any]] = [
                 {"role": "developer", "content": [{"type": "input_text", "text": base_instruction}]}
            ]

            # 2. 直前のセクションのHTMLを assistant メッセージとして追加 (もしあれば)
            if context.current_section_index > 0 and context.generated_sections_html:
                previous_section_html = context.generated_sections_html[-1]
                # MessageInputItem の代わりに辞書を使用
                current_input_messages.append(
                    {"role": "assistant", "content": [{"type": "output_text", "text": previous_section_html}]}
                )

            # 3. 今回の執筆依頼を user メッセージとして追加
            user_request = f"前のセクション（もしあれば）に続けて、アウトラインのセクション {context.current_section_index + 1}「{target_heading}」の内容をHTMLで執筆してください。"
            # MessageInputItem の代わりに辞書を使用
            current_input_messages.append(
                 {"role": "user", "content": [{"type": "input_text", "text": user_request}]}
            )

            agent_input = current_input_messages # Runnerにはリスト形式で渡す
            # ------------------------------------

            console.print(f"🤖 {current_agent.name} にセクション {context.current_section_index + 1} の執筆を依頼します (会話履歴利用)...")

        # --- 編集フェーズ ---
        elif context.current_step == "editing":
            # LiteLLM選択ロジック (変更なし)
            use_litellm = rich.prompt.Prompt.ask("編集にLiteLLMモデルを使用しますか？ (y/n)", choices=["y", "n"], default="n")
            if use_litellm.lower() == 'y' and LITELLM_AVAILABLE:
                litellm_model_name = rich.prompt.Prompt.ask("[cyan]使用するLiteLLMモデル名を入力してください (例: litellm/anthropic/claude-3-haiku-20240307)[/cyan]", default="litellm/anthropic/claude-3-haiku-20240307")
                # APIキーは環境変数から取得するか、ここでプロンプト表示するなど
                litellm_api_key = os.getenv(f"{litellm_model_name.split('/')[1].upper()}_API_KEY") # 例: ANTHROPIC_API_KEY
                if not litellm_api_key:
                     console.print(f"[yellow]警告: {litellm_model_name} のAPIキーが環境変数等で見つかりません。デフォルトのOpenAIモデルを使用します。[/yellow]")
                     current_agent = editor_agent
                else:
                     litellm_editor = get_litellm_agent("editor", litellm_model_name, litellm_api_key)
                     current_agent = litellm_editor if litellm_editor else editor_agent
            else:
                 current_agent = editor_agent

            agent_input = "記事ドラフト全体をレビューし、推敲・編集してください。"
            console.print(f"🤖 {current_agent.name} に最終編集を依頼します...")

        else:
            console.print(f"[bold red]未定義のステップ: {context.current_step}[/bold red]")
            context.current_step = "error"
            context.error_message = f"未定義のステップ {context.current_step} に到達しました。"
            continue

        # --- Agent実行 (リトライロジック追加) ---
        if not current_agent:
             context.error_message = "実行するエージェントが見つかりません。"
             context.current_step = "error"
             continue

        result = None
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                result = await Runner.run(
                    starting_agent=current_agent,
                    input=agent_input, # 文字列または会話履歴リスト
                    context=context,
                    run_config=run_config,
                    max_turns=10 # ツール使用等を考慮したターン数
                )
                # 成功したらループを抜ける
                last_exception = None
                break
            except InternalServerError as e:
                last_exception = e
                console.print(f"[yellow]サーバーエラー (500) が発生しました。リトライします... ({attempt + 1}/{MAX_RETRIES})[/yellow]")
                delay = INITIAL_RETRY_DELAY * (2 ** attempt) # Exponential backoff
                await asyncio.sleep(delay)
            except (MaxTurnsExceeded, ModelBehaviorError, BadRequestError, AgentsException, UserError) as e:
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
                import traceback
                traceback.print_exc()
                context.error_message = f"予期せぬエラー: {e}"
                context.current_step = "error"
                break # リトライせずにループを抜ける

        # リトライしてもエラーが解消しなかった場合
        if last_exception:
            if not context.error_message: # エラーメッセージがまだ設定されていなければ設定
                 context.error_message = f"Agent実行中にエラーが発生しました（リトライ上限到達）: {last_exception}"
            context.current_step = "error"
            continue # エラーステップへ

        # --- 結果処理 ---
        agent_output: Optional[AgentOutput] = None
        if result and result.final_output: # resultがNoneでないことを確認
             if isinstance(result.final_output, AgentOutput.__args__): # type: ignore
                  agent_output = result.final_output
             elif isinstance(result.final_output, str):
                  try:
                       parsed_output = json.loads(result.final_output)
                       agent_output = AgentOutput(**parsed_output) # type: ignore
                  except (json.JSONDecodeError, ValidationError) as parse_error:
                       console.print(f"[yellow]警告: Agentからの応答が予期したJSON形式ではありません。内容: {result.final_output[:100]}... エラー: {parse_error}[/yellow]")
                       agent_output = StatusUpdate(status="status_update", message=f"エージェントからの非構造応答: {result.final_output[:100]}...")

        context.last_agent_output = agent_output

        if not agent_output:
             console.print("[yellow]エージェントから有効な出力が得られませんでした。[/yellow]")
             context.error_message = "エージェントから有効な出力が得られませんでした。"
             context.current_step = "error"
             continue

        # --- ステップ更新 ---
        if isinstance(agent_output, ThemeProposal):
            context.current_step = "theme_proposed"
        elif isinstance(agent_output, ResearchPlan):
            context.research_plan = agent_output
            context.current_step = "research_plan_generated"
        elif isinstance(agent_output, ResearchQueryResult) and context.current_step == "researching": # 変更: ResearchQueryResult を処理
            # 現在のクエリと結果のクエリが一致するか確認（念のため）
            if context.research_plan and agent_output.query == context.research_plan.queries[context.current_research_query_index].query:
                context.add_query_result(agent_output) # 変更: add_snippet -> add_query_result
                console.print(f"[green]クエリ「{agent_output.query}」のリサーチ結果を処理しました。[/green]")
                context.current_research_query_index += 1 # 次のクエリへ
            else:
                 console.print(f"[yellow]警告: 予期しないクエリ「{agent_output.query}」の結果を受け取りました。[/yellow]")
                 context.error_message = "予期しないクエリの結果。"
                 context.current_step = "error"
            # researching ステップは継続
        elif isinstance(agent_output, ResearchReport):
            context.research_report = agent_output
            context.current_step = "research_report_generated"
        elif isinstance(agent_output, Outline):
            context.generated_outline = agent_output
            context.current_step = "outline_generated"
        elif isinstance(agent_output, ArticleSection):
            if agent_output.section_index == context.current_section_index:
                context.generated_sections_html.append(agent_output.html_content)
                console.print(f"[green]セクション {context.current_section_index + 1}「{agent_output.heading}」のHTMLが生成されました。[/green]")
                display_article_preview(agent_output.html_content, f"セクション {context.current_section_index + 1} プレビュー")
                context.current_section_index += 1
                # writing_sections ステップは継続
            else:
                console.print(f"[yellow]警告: 予期しないセクションインデックス {agent_output.section_index} の応答を受け取りました（期待値: {context.current_section_index}）。[/yellow]")
                context.error_message = "予期しないセクションインデックスの応答。"
                context.current_step = "error"
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
             # StatusUpdateを受け取った場合のステップ遷移ロジックが必要な場合がある

    # --- ループ終了後 ---
    if context.current_step == "completed":
        console.print("\n🎉 [bold green]SEO記事の生成が正常に完了しました。[/bold green]")
        if context.final_article_html:
             save_confirm = rich.prompt.Prompt.ask("最終記事を 'final_article.html' として保存しますか？ (y/n)", choices=["y", "n"], default="y")
             if save_confirm.lower() == 'y':
                  save_article(context.final_article_html, "final_article.html")
        else:
             console.print("[yellow]警告: 最終記事コンテンツが見つかりません。[/yellow]")

    elif context.current_step == "error":
        console.print(f"\n❌ [bold red]記事生成プロセス中にエラーが発生しました。[/bold red]")
        if context.error_message:
            console.print(f"エラー詳細: {context.error_message}")

    console.print("プログラムを終了します。")


async def main():
    console.print("[bold magenta]📝 SEO記事生成システム (リサーチ強化版) へようこそ！[/bold magenta]")

    # --- ユーザーからの初期情報入力 ---
    keywords_str = rich.prompt.Prompt.ask("[cyan]ターゲットキーワードを入力してください（カンマ区切り）[/cyan]", default="芝生, 育て方, 初心者")
    initial_keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

    target_persona = rich.prompt.Prompt.ask("[cyan]ターゲットペルソナを入力してください（例：庭づくり初心者, 子供がいる家庭）[/cyan]", default="庭づくり初心者")

    target_length_str = rich.prompt.Prompt.ask("[cyan]目標文字数を入力してください（任意、数値のみ）[/cyan]", default="2000")
    target_length = None
    if target_length_str.isdigit():
        target_length = int(target_length_str)

    num_themes_str = rich.prompt.Prompt.ask("[cyan]提案してほしいテーマ数を入力してください[/cyan]", default="3")
    num_theme_proposals = 3
    if num_themes_str.isdigit() and int(num_themes_str) > 0:
         num_theme_proposals = int(num_themes_str)

    num_research_queries_str = rich.prompt.Prompt.ask("[cyan]リサーチで使用する検索クエリ数を入力してください[/cyan]", default="3") # デフォルトを3に減らしてテスト
    num_research_queries = 3
    if num_research_queries_str.isdigit() and int(num_research_queries_str) > 0:
        num_research_queries = int(num_research_queries_str)

    vector_store_id = rich.prompt.Prompt.ask("[cyan]File Searchで使用するVector Store IDを入力してください（任意）[/cyan]", default="")


    # --- コンテキスト初期化 ---
    article_context = ArticleContext(
        initial_keywords=initial_keywords,
        target_persona=target_persona,
        target_length=target_length,
        num_theme_proposals=num_theme_proposals,
        num_research_queries=num_research_queries,
        vector_store_id=vector_store_id if vector_store_id else None,
    )

    # --- 実行設定 ---
    run_config = RunConfig(
        workflow_name="SEOArticleGenerationWithResearch",
        trace_id=f"trace_{uuid.uuid4().hex}",
    )

    # --- メインループ実行 ---
    await run_main_loop(article_context, run_config)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n[bold red]プログラム実行中に致命的なエラーが発生しました: {e}[/bold red]")
        import traceback
        traceback.print_exc()
