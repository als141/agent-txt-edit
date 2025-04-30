# -*- coding: utf-8 -*-
"""
SEO最適化記事生成システム
OpenAI Agents SDKを使用した記事自動生成ツール
"""

import os
import json
import asyncio
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Literal, Tuple, TypedDict, cast
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.progress import Progress
from rich.syntax import Syntax
from bs4 import BeautifulSoup
import requests

# OpenAI Agents SDK
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
    # トレーシング
    trace,
    # 動的プロンプト用
)
# ウェブ検索ツール
from agents.tool import WebSearchTool
# Modelプロバイダー
from agents.models.openai_responses import OpenAIResponsesModel

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 環境変数の読み込み ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI APIキーが設定されていません。.envファイルを確認してください。")

# --- グローバル設定 ---
console = Console()
# デフォルトモデル設定
DEFAULT_MODEL = "gpt-4o-mini"
# より高品質な記事生成に使用するモデル
PREMIUM_MODEL = "gpt-4o"

# --- Pydanticモデル ---
class CompanyInfo(BaseModel):
    """企業情報を格納するモデル"""
    name: str = Field(description="企業名")
    industry: str = Field(description="業界")
    target_audience: str = Field(description="ターゲットオーディエンス")
    tone: str = Field(description="文体・トーン", default="普通")
    website: Optional[str] = Field(description="Webサイト URL", default=None)
    previous_articles: List[str] = Field(description="過去の記事タイトルリスト", default_factory=list)
    
    # 追加プロパティを禁止する設定
    model_config = ConfigDict(extra="forbid")

class ArticleTopic(BaseModel):
    """記事トピックの提案"""
    title: str = Field(description="記事タイトル")
    description: str = Field(description="概要説明")
    keywords: List[str] = Field(description="関連キーワード")
    rationale: str = Field(description="この提案理由")
    
    model_config = ConfigDict(extra="forbid")

class SectionInfo(BaseModel):
    """記事のセクション情報"""
    heading: str = Field(description="見出し")
    description: str = Field(description="説明")
    
    model_config = ConfigDict(extra="forbid")

class ArticleStructure(BaseModel):
    """記事構造の定義"""
    title: str = Field(description="記事タイトル")
    sections: List[SectionInfo] = Field(description="セクション構成")
    tone: str = Field(description="文体・トーン")
    target_word_count: int = Field(description="目標総文字数")
    target_audience: str = Field(description="ターゲット読者")
    keywords: List[str] = Field(description="主要キーワード")
    
    model_config = ConfigDict(extra="forbid")

class ArticleSection(BaseModel):
    """記事セクションの内容"""
    heading: str = Field(description="見出し")
    content: str = Field(description="HTML形式のコンテンツ")
    word_count: int = Field(description="文字数")
    
    model_config = ConfigDict(extra="forbid")

class KeywordDensity(BaseModel):
    """キーワード密度情報"""
    density: float = Field(description="密度 (%)")
    
    model_config = ConfigDict(extra="forbid")

class SEOAnalysis(BaseModel):
    """SEO分析結果"""
    keyword_density: Dict[str, float] = Field(description="キーワード密度")
    readability_score: float = Field(description="読みやすさスコア (0-100)")
    issues: List[str] = Field(description="SEO上の問題点")
    recommendations: List[str] = Field(description="改善提案")
    
    model_config = ConfigDict(extra="forbid")

class EditOperation(BaseModel):
    """編集操作"""
    section_index: int = Field(description="編集対象のセクションインデックス")
    original_text: str = Field(description="元のテキスト")
    new_text: str = Field(description="新しいテキスト")
    explanation: str = Field(description="編集理由")
    
    model_config = ConfigDict(extra="forbid")

class CompetitorAnalysis(BaseModel):
    """競合記事の分析結果"""
    title: str = Field(description="記事タイトル")
    url: str = Field(description="URL")
    main_points: List[str] = Field(description="主要なポイント")
    keywords: List[str] = Field(description="使用されているキーワード")
    structure: List[str] = Field(description="記事構造")
    strengths: List[str] = Field(description="強み")
    weaknesses: List[str] = Field(description="弱み")
    
    model_config = ConfigDict(extra="forbid")

class CompetitorArticle(BaseModel):
    """競合記事の基本情報"""
    title: str = Field(description="記事タイトル")
    url: str = Field(description="URL")
    snippet: str = Field(description="記事の概要")
    published_date: str = Field(description="公開日")
    
    model_config = ConfigDict(extra="forbid")

class WordDistribution(BaseModel):
    """セクションごとの文字数配分"""
    heading: str = Field(description="見出し")
    description: str = Field(description="説明")
    target_word_count: int = Field(description="目標文字数")
    
    model_config = ConfigDict(extra="forbid")

class SectionForHTML(BaseModel):
    """HTML形式への変換用セクション情報"""
    heading: str = Field(description="見出し")
    content: str = Field(description="HTML内容")
    
    model_config = ConfigDict(extra="forbid")

class KeywordAnalysisResult(BaseModel):
    """キーワード分析結果"""
    search_volume: int = Field(description="検索ボリューム")
    competition: float = Field(description="競合度")
    cpc: float = Field(description="クリック単価")
    suggested_related_keywords: List[str] = Field(description="関連キーワード提案")
    
    model_config = ConfigDict(extra="forbid")

class SEOKeywordAnalysis(BaseModel):
    """SEOキーワード分析の結果"""
    status: str = Field(description="ステータス")
    data: Dict[str, KeywordAnalysisResult] = Field(description="キーワードごとのデータ")
    timestamp: float = Field(description="タイムスタンプ")
    
    model_config = ConfigDict(extra="forbid")

class ArticleStructureAnalysis(BaseModel):
    """記事構造の分析結果"""
    title: str = Field(description="記事タイトル")
    headings: List[str] = Field(description="見出し")
    estimated_word_count: int = Field(description="推定文字数")
    main_keywords: List[str] = Field(description="主要キーワード")
    has_images: bool = Field(description="画像の有無")
    has_videos: bool = Field(description="動画の有無")
    has_internal_links: bool = Field(description="内部リンクの有無")
    has_external_links: bool = Field(description="外部リンクの有無")
    
    model_config = ConfigDict(extra="forbid")

# --- コンテキストクラス ---
@dataclass
class ArticleGenerationContext:
    """記事生成コンテキスト"""
    # 企業情報
    company_info: Optional[CompanyInfo] = None
    # 記事トピック
    topics: List[ArticleTopic] = field(default_factory=list)
    selected_topic: Optional[ArticleTopic] = None
    # 記事構造
    article_structures: List[ArticleStructure] = field(default_factory=list)
    selected_structure: Optional[ArticleStructure] = None
    # 記事セクション
    sections: List[ArticleSection] = field(default_factory=list)
    # SEO分析
    seo_analysis: Optional[SEOAnalysis] = None
    # 競合分析
    competitor_analyses: List[CompetitorAnalysis] = field(default_factory=list)
    # 編集操作履歴
    edit_history: List[EditOperation] = field(default_factory=list)
    # 生成設定
    generation_settings: Dict[str, Any] = field(default_factory=dict)
    # 現在の状態
    current_state: str = "初期化"
    # 完成した記事のHTML
    final_article_html: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """コンテキストを辞書に変換"""
        return {
            "company_info": self.company_info.model_dump() if self.company_info else None,
            "topics": [topic.model_dump() for topic in self.topics],
            "selected_topic": self.selected_topic.model_dump() if self.selected_topic else None,
            "article_structures": [structure.model_dump() for structure in self.article_structures],
            "selected_structure": self.selected_structure.model_dump() if self.selected_structure else None,
            "sections": [section.model_dump() for section in self.sections],
            "seo_analysis": self.seo_analysis.model_dump() if self.seo_analysis else None,
            "competitor_analyses": [analysis.model_dump() for analysis in self.competitor_analyses],
            "edit_history": [edit.model_dump() for edit in self.edit_history],
            "generation_settings": self.generation_settings,
            "current_state": self.current_state,
            "final_article_html": self.final_article_html,
        }
    
    def get_current_article_html(self) -> str:
        """現在のセクションを連結してHTML記事を生成"""
        if not self.sections:
            return "<article><p>記事はまだ生成されていません。</p></article>"
        
        html = "<article>\n"
        for section in self.sections:
            html += section.content + "\n"
        html += "</article>"
        return html
    
    def get_total_word_count(self) -> int:
        """現在の記事の総文字数を取得"""
        return sum(section.word_count for section in self.sections)
    
    def get_remaining_sections(self) -> List[SectionInfo]:
        """まだ生成されていないセクションを取得"""
        if not self.selected_structure:
            return []
        
        generated_headings = {section.heading for section in self.sections}
        remaining_sections = []
        
        for section in self.selected_structure.sections:
            if section.heading not in generated_headings:
                remaining_sections.append(section)
                
        return remaining_sections

# --- ツール定義 ---
@function_tool
async def analyze_seo_keywords(
    ctx: RunContextWrapper[ArticleGenerationContext], 
    keywords: List[str]
) -> SEOKeywordAnalysis:
    """
    指定されたキーワードの検索ボリュームとSEO難易度を分析します。
    
    Args:
        keywords: 分析するキーワードのリスト
    
    Returns:
        SEOKeywordAnalysis: 各キーワードのSEO情報を含む分析結果
    """
    # 実際のAPIを使用する場合はここで呼び出し
    # この例ではモックデータを返す
    console.print(f"[dim]SEOキーワード分析中: {', '.join(keywords)}[/dim]")
    time.sleep(1)  # API呼び出しの遅延をシミュレート
    
    results = {}
    for keyword in keywords:
        # ランダム値の代わりに実際のAPIからデータを取得
        search_volume = hash(keyword) % 10000
        competition = (hash(keyword) % 100) / 100
        cpc = (hash(keyword) % 500) / 100
        results[keyword] = KeywordAnalysisResult(
            search_volume=search_volume,
            competition=competition,
            cpc=cpc,
            suggested_related_keywords=[
                f"{keyword} とは",
                f"{keyword} 方法",
                f"最新 {keyword}",
                f"{keyword} 例"
            ]
        )
    
    return SEOKeywordAnalysis(
        status="success",
        data=results,
        timestamp=time.time()
    )

@function_tool
async def analyze_article_seo(
    ctx: RunContextWrapper[ArticleGenerationContext], 
    html_content: str,
    target_keywords: List[str]
) -> SEOAnalysis:
    """
    HTMLコンテンツのSEO分析を行います。
    
    Args:
        html_content: 分析するHTML形式の記事内容
        target_keywords: ターゲットキーワードのリスト
    
    Returns:
        SEOAnalysis: SEO分析結果
    """
    console.print("[dim]記事のSEO分析中...[/dim]")
    
    # HTMLからプレーンテキストを抽出
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    word_count = len(text_content)
    
    # キーワード密度の計算
    keyword_density = {}
    for keyword in target_keywords:
        count = text_content.lower().count(keyword.lower())
        density = count / (word_count / 100) if word_count > 0 else 0
        keyword_density[keyword] = round(density, 2)
    
    # 見出しの確認
    headings = [h.text for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
    heading_keywords = sum(1 for h in headings for k in target_keywords if k.lower() in h.lower())
    
    # 内部リンクのカウント
    internal_links = len(soup.find_all('a', href=True))
    
    # 画像の代替テキストチェック
    images = soup.find_all('img')
    images_with_alt = sum(1 for img in images if img.get('alt'))
    
    # パラグラフの長さチェック
    paragraphs = soup.find_all('p')
    avg_paragraph_length = sum(len(p.text) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    # 読みやすさスコアの計算 (簡易版)
    long_sentences = sum(1 for p in paragraphs for s in p.text.split('.') if len(s) > 100)
    readability_score = 100 - (long_sentences * 5) - (avg_paragraph_length > 400) * 20
    readability_score = max(0, min(100, readability_score))
    
    # 問題点の特定
    issues = []
    if heading_keywords < len(target_keywords) / 2:
        issues.append("見出しにキーワードが十分に含まれていません")
    
    if any(density > 5 for density in keyword_density.values()):
        issues.append("キーワードの過剰使用（キーワードスタッフィング）の可能性があります")
    
    if any(density < 0.5 for density in keyword_density.values()):
        issues.append("主要キーワードの使用頻度が低すぎます")
    
    if internal_links < 2:
        issues.append("内部リンクが少なすぎます")
    
    if images and images_with_alt < len(images):
        issues.append("一部の画像にalt属性がありません")
    
    if avg_paragraph_length > 400:
        issues.append("段落が長すぎます。読みやすさのために分割を検討してください")
    
    # 改善提案
    recommendations = []
    if heading_keywords < len(target_keywords) / 2:
        recommendations.append("見出しにもっとキーワードを含めてください")
    
    if any(density > 5 for density in keyword_density.values()):
        recommendations.append("キーワードの使用を自然にし、過剰使用を避けてください")
    
    if any(density < 0.5 for density in keyword_density.values()):
        recommendations.append("コンテンツ内でターゲットキーワードをもっと自然に使用してください")
    
    if internal_links < 2:
        recommendations.append("関連コンテンツへの内部リンクを追加してください")
    
    if images and images_with_alt < len(images):
        recommendations.append("すべての画像に適切なalt属性を追加してください")
    
    if avg_paragraph_length > 400:
        recommendations.append("長い段落を分割し、読みやすくしてください")
    
    return SEOAnalysis(
        keyword_density=keyword_density,
        readability_score=round(readability_score, 1),
        issues=issues,
        recommendations=recommendations
    )

@function_tool
async def get_competitor_articles(
    ctx: RunContextWrapper[ArticleGenerationContext],
    topic: str,
    keywords: List[str],
    num_results: int
) -> List[CompetitorArticle]:
    """
    指定されたトピックと関連キーワードに基づいて競合記事を検索します。
    実際の実装ではウェブ検索ツールを使用します。
    
    Args:
        topic: 検索するトピック
        keywords: 関連キーワードのリスト
        num_results: 取得する結果の数
    
    Returns:
        List[CompetitorArticle]: 競合記事のリスト
    """
    console.print(f"[dim]競合記事を検索中: {topic}[/dim]")
    
    # 実際にはWebSearchToolを使用してウェブ検索を行います
    # このモック実装では、ダミーデータを返します
    search_query = f"{topic} {' '.join(keywords[:3])}"
    
    # 実際のAPIレスポンスを模倣したダミーデータ
    results = []
    for i in range(min(num_results, 5)):
        results.append(CompetitorArticle(
            title=f"{topic}に関する包括的ガイド {i+1}",
            url=f"https://example.com/blog/{i+1}",
            snippet=f"{topic}についての詳細解説。{', '.join(keywords[:2])}に焦点を当て、実践的なアドバイスを提供します。",
            published_date=f"2024-{(i+1):02d}-15"
        ))
    
    return results

@function_tool
async def extract_article_structure(
    ctx: RunContextWrapper[ArticleGenerationContext],
    url: str
) -> ArticleStructureAnalysis:
    """
    指定されたURLから記事の構造を抽出します。
    
    Args:
        url: 分析する記事のURL
    
    Returns:
        ArticleStructureAnalysis: 記事の構造情報
    """
    console.print(f"[dim]記事構造を分析中: {url}[/dim]")
    
    # 実際の実装ではURLから記事を取得して解析します
    # このモック実装では、ダミーデータを返します
    
    # URLからハッシュ値を生成して一貫性のあるダミーデータを生成
    hash_value = hash(url)
    
    headings = [
        "はじめに",
        "主要なポイント",
        "実践的なステップ",
        "よくある間違い",
        "ケーススタディ",
        "まとめ"
    ]
    
    # 異なるURLに対して少し異なる構造を返す
    selected_headings = headings[hash_value % 3:hash_value % 3 + 4]
    
    return ArticleStructureAnalysis(
        title=f"分析した記事のタイトル ({url})",
        headings=selected_headings,
        estimated_word_count=1200 + (hash_value % 800),
        main_keywords=[
            f"キーワード{hash_value % 10 + 1}",
            f"キーワード{hash_value % 5 + 6}",
            f"キーワード{hash_value % 7 + 3}"
        ],
        has_images=bool(hash_value % 2),
        has_videos=bool(hash_value % 3),
        has_internal_links=bool(hash_value % 2),
        has_external_links=True
    )

@function_tool
async def generate_html_section(
    ctx: RunContextWrapper[ArticleGenerationContext],
    heading: str,
    content_brief: str,
    target_word_count: int,
    keywords: List[str],
    tone: str
) -> ArticleSection:
    """
    指定された見出しと要約に基づいてHTML形式のセクションを生成します。
    
    Args:
        heading: セクションの見出し
        content_brief: セクションの内容概要
        target_word_count: 目標文字数
        keywords: 含めるべきキーワード
        tone: 文体やトーン
    
    Returns:
        ArticleSection: 生成されたセクション
    """
    # 実際の実装ではAIを利用してコンテンツを生成します
    # このツールはAPI呼び出しのプレースホルダーです
    console.print(f"[dim]セクション '{heading}' ({target_word_count}文字) を生成中...[/dim]")
    
    # 実際のHTMLコンテンツ生成は別のエージェントで行います
    # ここでは仮のHTMLを返しておきます
    html_content = f"""<section>
  <h2>{heading}</h2>
  <p>このセクションには {content_brief} についての内容が含まれます。ここには約 {target_word_count} 文字の文章が生成されます。</p>
  <p>キーワード: {', '.join(keywords[:3])}</p>
  <p>トーン: {tone}</p>
</section>"""
    
    return ArticleSection(
        heading=heading,
        content=html_content,
        word_count=len(html_content)
    )

@function_tool
async def calculate_word_distribution(
    ctx: RunContextWrapper[ArticleGenerationContext],
    total_word_count: int,
    sections: List[SectionInfo]
) -> List[WordDistribution]:
    """
    総文字数を各セクションに分配します。
    
    Args:
        total_word_count: 総文字数
        sections: セクションのリスト
    
    Returns:
        List[WordDistribution]: 文字数が割り当てられたセクションのリスト
    """
    console.print(f"[dim]文字数分配計算中: 合計 {total_word_count} 文字, {len(sections)} セクション[/dim]")
    
    result = []
    
    # 重要度に基づいてセクションに文字数を分配
    # 導入部と結論は少し短く、メインコンテンツに多くの文字数を割り当てる
    
    # セクション数に基づく基本配分
    base_count_per_section = total_word_count / len(sections)
    
    for i, section in enumerate(sections):
        heading = section.heading
        description = section.description
        
        # 重要度係数の決定（導入と結論はやや短く）
        importance = 1.0  # デフォルト
        if i == 0 or "はじめに" in heading or "導入" in heading:
            importance = 0.8  # 導入部は少し短く
        elif i == len(sections) - 1 or "まとめ" in heading or "結論" in heading:
            importance = 0.7  # 結論部分も少し短く
        elif "主要" in heading or "重要" in heading or "核心" in heading:
            importance = 1.2  # 主要部分は少し長く
        
        # 文字数の計算
        word_count = int(base_count_per_section * importance)
        
        result.append(WordDistribution(
            heading=heading,
            description=description,
            target_word_count=word_count
        ))
    
    # 合計が目標文字数になるように調整
    current_total = sum(item.target_word_count for item in result)
    adjustment = total_word_count - current_total
    
    # 最も長いセクションに調整分を追加
    if adjustment != 0:
        max_section_idx = max(range(len(result)), key=lambda i: result[i].target_word_count)
        result[max_section_idx].target_word_count += adjustment
    
    return result

@function_tool
async def format_article_to_html(
    ctx: RunContextWrapper[ArticleGenerationContext],
    title: str,
    sections: List[SectionForHTML],
    meta_description: str = ""
) -> str:
    """
    記事のセクションをHTML形式に整形します。
    
    Args:
        title: 記事のタイトル
        sections: セクションのリスト
        meta_description: メタディスクリプション
    
    Returns:
        str: 完全なHTML記事
    """
    console.print(f"[dim]HTML形式に整形中: {title}[/dim]")
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <meta name="description" content="{meta_description}">
</head>
<body>
    <article>
        <h1>{title}</h1>
"""
    
    for section in sections:
        html += f"""
        {section.content}
"""
    
    html += """
    </article>
</body>
</html>"""
    
    return html

# --- 動的プロンプト生成関数 ---
def create_dynamic_instructions(base_prompt: str) -> callable:
    async def dynamic_instructions_func(ctx: RunContextWrapper[ArticleGenerationContext], agent: Agent[ArticleGenerationContext]) -> str:
        context_info = ""
        
        if ctx.context.company_info:
            company_info = ctx.context.company_info
            context_info += f"""
【企業情報】
企業名: {company_info.name}
業界: {company_info.industry}
ターゲット: {company_info.target_audience}
文体・トーン: {company_info.tone}
"""
            if company_info.website:
                context_info += f"Webサイト: {company_info.website}\n"
            if company_info.previous_articles:
                context_info += f"過去の記事: {', '.join(company_info.previous_articles[:5])}\n"

        if ctx.context.selected_topic:
            topic = ctx.context.selected_topic
            context_info += f"""
【選択されたトピック】
タイトル: {topic.title}
説明: {topic.description}
キーワード: {', '.join(topic.keywords)}
"""

        if ctx.context.selected_structure:
            structure = ctx.context.selected_structure
            context_info += f"""
【記事構造】
タイトル: {structure.title}
文体・トーン: {structure.tone}
目標文字数: {structure.target_word_count}
ターゲット読者: {structure.target_audience}
キーワード: {', '.join(structure.keywords)}

セクション:
"""
            for i, section in enumerate(structure.sections):
                context_info += f"{i+1}. {section.heading}: {section.description}\n"

        if ctx.context.sections:
            context_info += f"""
【生成済みセクション】
"""
            for i, section in enumerate(ctx.context.sections):
                context_info += f"{i+1}. {section.heading} ({section.word_count}文字)\n"

        if ctx.context.seo_analysis:
            seo = ctx.context.seo_analysis
            context_info += f"""
【SEO分析】
読みやすさスコア: {seo.readability_score}/100
主な問題点: {', '.join(seo.issues[:3])}
推奨改善: {', '.join(seo.recommendations[:3])}
"""

        if ctx.context.competitor_analyses:
            context_info += f"""
【競合分析】
分析済み競合記事数: {len(ctx.context.competitor_analyses)}
"""

        context_info += f"""
【現在の状態】
状態: {ctx.context.current_state}
生成済みセクション数: {len(ctx.context.sections)}
"""

        if ctx.context.selected_structure and ctx.context.sections:
            remaining = len(ctx.context.selected_structure.sections) - len(ctx.context.sections)
            context_info += f"残りセクション数: {remaining}\n"
            if ctx.context.selected_structure.target_word_count > 0:
                current = ctx.context.get_total_word_count()
                remaining_words = ctx.context.selected_structure.target_word_count - current
                context_info += f"現在の文字数: {current}/{ctx.context.selected_structure.target_word_count} (残り: {remaining_words})\n"

        full_prompt = f"""{base_prompt}

--- 追加コンテキスト情報 ---
{context_info}
--- 追加コンテキスト情報終 ---
"""
        return full_prompt
    
    return dynamic_instructions_func

# --- エージェント定義 ---
# リサーチエージェント
RESEARCH_AGENT_PROMPT = """
あなたは記事作成のためのリサーチを専門とするエージェントです。
与えられたテーマや業界について、適切な情報収集と分析を行います。

**主な責務:**
1. 指定されたテーマについてウェブ検索を行い、関連情報を収集する
2. 競合記事を分析し、その構造や特徴を把握する
3. SEOキーワードの分析と提案を行う
4. 収集した情報に基づき、記事トピックや構成の提案を行う

**出力形式:**
- 検索結果のサマリー
- キーワード分析結果
- 競合分析レポート
- 記事トピックの提案リスト

ユーザーの指示に応じて、適切なツールを利用してリサーチを実行し、結果をわかりやすく構造化して返してください。
"""

# プランニングエージェント
PLANNING_AGENT_PROMPT = """
あなたは記事構成の設計を専門とするエージェントです。
SEO最適化された記事の全体構造と詳細な各セクションの設計を担当します。

**主な責務:**
1. リサーチ結果に基づいた記事構造の設計
2. 各セクションの見出しとコンテンツ概要の作成
3. SEOキーワードの適切な配置計画
4. ターゲット読者と意図に合わせた構成の最適化

**出力形式:**
- 完全な記事構造プラン
- 文字数配分の提案
- SEOキーワード配置戦略
- 各セクションの詳細な概要

プランニングの際は以下の点を考慮してください:
- 読者の関心を引く導入部
- 論理的な情報の流れ
- 適切な見出し階層構造
- 効果的なCTAの配置
- SEOに最適化された構造

リサーチエージェントからの情報とユーザーの要望に基づき、明確で効果的な記事構造を設計してください。
"""

# 執筆エージェント
WRITING_AGENT_PROMPT = """
あなたは記事の執筆を専門とするエージェントです。
設計された構造に従って、高品質でSEO最適化された記事コンテンツを生成します。

**主な責務:**
1. 各セクションのHTML形式コンテンツ生成
2. 自然な形でのSEOキーワードの組み込み
3. 読みやすく魅力的な文章の作成
4. 指定された文体やトーンに合わせた執筆
5. 指定の文字数制限内での執筆

**重要な指示:**
- HTMLタグを適切に使用し、有効なHTMLを生成すること
- 指定されたキーワードを自然に組み込むこと
- 読者の興味を引く導入部を作成すること
- 事実に基づいた正確な情報を提供すること
- 指定された文体やトーンを一貫して維持すること
- 各セクションは個別に生成し、指定された文字数に合わせること
- HTMLのみを出力し、それ以外の説明やコメントを含めないこと

**HTML形式の例:**
```html
<section>
  <h2>セクションの見出し</h2>
  <p>段落のテキスト...</p>
  <ul>
    <li>リストアイテム1</li>
    <li>リストアイテム2</li>
  </ul>
  <p>さらなるテキスト...</p>
</section>
```

プランニングエージェントが設計した構造に忠実に従いながら、クリエイティブで価値のあるコンテンツを生成してください。
"""

# 編集エージェント
EDITING_AGENT_PROMPT = """
あなたは記事の編集と最適化を専門とするエージェントです。
生成された記事を精査し、SEO、読みやすさ、正確性の観点から改善を行います。

**主な責務:**
1. 記事のSEO分析と最適化提案
2. 文法・表現・論理構成の確認と修正
3. キーワードの適切な配置と密度の調整
4. 読みやすさとユーザーエクスペリエンスの向上
5. 最終的なHTML形式の調整と整形

**編集の際に確認すべき点:**
- キーワードが自然に組み込まれているか
- 文章の流れが論理的で一貫しているか
- 段落の長さは適切か
- 見出しが明確で階層構造が適切か
- 誤字脱字や文法ミスがないか
- 内部リンク・外部リンクの配置は適切か
- 画像のalt属性は適切か
- メタディスクリプションは魅力的で適切な長さか

記事を精査し、具体的な改善提案を行うとともに、必要な修正を加えた最終バージョンを提供してください。
"""

# トリアージエージェント
TRIAGE_AGENT_PROMPT = """
あなたはSEO最適化記事生成システムの司令塔エージェントです。
ユーザーからの指示を分析し、最適なエージェントに作業を振り分けます。

**メインエージェント:**
1. リサーチエージェント: 情報収集と競合分析を担当
2. プランニングエージェント: 記事構造の設計を担当
3. 執筆エージェント: 実際の記事コンテンツ生成を担当
4. 編集エージェント: 記事の編集と最終調整を担当

**あなたの主な責務:**
1. ユーザーの指示を理解し、どのエージェントに振り分けるべきか判断する
2. 必要に応じて追加情報をユーザーに質問する
3. 適切なエージェントにハンドオフする
4. エージェントからの結果をユーザーに説明する
5. 全体のワークフローを管理する

**プロセスフロー:**
1. 初期ユーザー情報の収集 (企業情報、ターゲット、既存の記事など)
2. トピック提案のためのリサーチ実施
3. ユーザーによるトピック選択
4. 記事構造の設計と提案
5. ユーザーによる構造確認
6. セクション単位での記事生成
7. 記事の編集と最終調整
8. 完成した記事の納品

ユーザーの指示に応じて、適切なエージェントにハンドオフするか、自身で応答するかを判断してください。
現在のワークフロー状態に応じて次のステップを提案し、プロセスを前進させることを心がけてください。
"""

# エージェントの作成
def create_agents():
    """エージェントを作成して返す"""
    # トリアージエージェント（メインエージェント）
    web_search_tool = WebSearchTool()
    
    # リサーチエージェント
    research_agent = Agent[ArticleGenerationContext](
        name="リサーチエージェント",
        handoff_description="記事作成のためのリサーチを担当するエージェント。ウェブ検索、競合分析、SEOキーワード分析を行います。",
        instructions=create_dynamic_instructions(RESEARCH_AGENT_PROMPT),
        model=DEFAULT_MODEL,
        tools=[
            web_search_tool,
            analyze_seo_keywords,
            get_competitor_articles,
            extract_article_structure,
        ],
    )
    
    # プランニングエージェント
    planning_agent = Agent[ArticleGenerationContext](
        name="プランニングエージェント",
        handoff_description="記事構造の設計を担当するエージェント。記事の全体構造と各セクションの詳細を設計します。",
        instructions=create_dynamic_instructions(PLANNING_AGENT_PROMPT),
        model=DEFAULT_MODEL,
        tools=[
            calculate_word_distribution,
        ],
    )
    
    # 執筆エージェント
    writing_agent = Agent[ArticleGenerationContext](
        name="執筆エージェント",
        handoff_description="記事の執筆を担当するエージェント。HTML形式でセクションごとに記事を生成します。",
        instructions=create_dynamic_instructions(WRITING_AGENT_PROMPT),
        model=PREMIUM_MODEL,  # 高品質なコンテンツ生成にはより高性能なモデルを使用
        model_settings=ModelSettings(
            temperature=0.7,  # より創造的な文章のために温度を上げる
        ),
        tools=[
            generate_html_section,
        ],
    )
    
    # 編集エージェント
    editing_agent = Agent[ArticleGenerationContext](
        name="編集エージェント",
        handoff_description="記事の編集と最適化を担当するエージェント。SEO、読みやすさ、正確性の観点から記事を改善します。",
        instructions=create_dynamic_instructions(EDITING_AGENT_PROMPT),
        model=DEFAULT_MODEL,
        tools=[
            analyze_article_seo,
            format_article_to_html,
        ],
    )
    
    # トリアージエージェント
    triage_agent = Agent[ArticleGenerationContext](
        name="トリアージエージェント",
        instructions=create_dynamic_instructions(TRIAGE_AGENT_PROMPT),
        model=DEFAULT_MODEL,
        tools=[],
        handoffs=[research_agent, planning_agent, writing_agent, editing_agent],
    )
    
    return {
        "triage_agent": triage_agent,
        "research_agent": research_agent,
        "planning_agent": planning_agent,
        "writing_agent": writing_agent,
        "editing_agent": editing_agent
    }

# --- ヘルパー関数 ---
def get_company_info() -> CompanyInfo:
    """ユーザーから企業情報を収集する"""
    console.print(Panel.fit("[bold cyan]企業情報入力[/bold cyan]", title="Step 1"))
    
    name = Prompt.ask("[bold]企業名[/bold]")
    industry = Prompt.ask("[bold]業界[/bold]")
    target_audience = Prompt.ask("[bold]ターゲットオーディエンス[/bold]")
    tone = Prompt.ask("[bold]文体・トーン[/bold]", default="普通")
    website = Prompt.ask("[bold]Webサイト URL[/bold] (任意)", default="")
    
    previous_articles_str = Prompt.ask("[bold]過去の記事タイトル[/bold] (カンマ区切りで入力、任意)", default="")
    previous_articles = [a.strip() for a in previous_articles_str.split(",") if a.strip()]
    
    return CompanyInfo(
        name=name,
        industry=industry,
        target_audience=target_audience,
        tone=tone,
        website=website if website else None,
        previous_articles=previous_articles
    )

def display_topic_proposals(topics: List[ArticleTopic]) -> int:
    """トピック提案を表示し、ユーザーに選択させる"""
    console.print(Panel.fit("[bold cyan]記事トピック提案[/bold cyan]", title="Step 2"))
    
    for i, topic in enumerate(topics):
        console.print(f"[bold]{i+1}. {topic.title}[/bold]")
        console.print(f"  説明: {topic.description}")
        console.print(f"  キーワード: {', '.join(topic.keywords)}")
        console.print(f"  提案理由: {topic.rationale}")
        console.print("")
    
    while True:
        choice = Prompt.ask("[bold]選択するトピック番号を入力してください[/bold]", default="1")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(topics):
                return choice_idx
            else:
                console.print("[red]無効な番号です。もう一度入力してください。[/red]")
        except ValueError:
            console.print("[red]数値を入力してください。[/red]")

def display_structure_proposals(structures: List[ArticleStructure]) -> int:
    """記事構造の提案を表示し、ユーザーに選択させる"""
    console.print(Panel.fit("[bold cyan]記事構造提案[/bold cyan]", title="Step 3"))
    
    for i, structure in enumerate(structures):
        console.print(f"[bold]{i+1}. {structure.title}[/bold]")
        console.print(f"  文体・トーン: {structure.tone}")
        console.print(f"  総文字数: {structure.target_word_count}")
        console.print(f"  ターゲット: {structure.target_audience}")
        console.print(f"  キーワード: {', '.join(structure.keywords)}")
        console.print("  セクション:")
        for j, section in enumerate(structure.sections):
            console.print(f"    {j+1}. {section.heading}: {section.description}")
        console.print("")
    
    while True:
        choice = Prompt.ask("[bold]選択する構造番号を入力してください[/bold]", default="1")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(structures):
                return choice_idx
            else:
                console.print("[red]無効な番号です。もう一度入力してください。[/red]")
        except ValueError:
            console.print("[red]数値を入力してください。[/red]")

def display_article_preview(html_content: str) -> None:
    """記事のプレビューを表示する"""
    console.print(Panel.fit("[bold cyan]記事プレビュー[/bold cyan]", title="Preview"))
    
    # HTMLから見出しと本文を抽出して表示
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # タイトル(h1)の抽出
    title = soup.find('h1')
    if title:
        console.print(f"[bold cyan]{title.text}[/bold cyan]")
    
    # 各セクションの表示
    for heading in soup.find_all(['h2', 'h3']):
        level = 2 if heading.name == 'h2' else 3
        indent = "  " * (level - 2)
        console.print(f"\n{indent}[bold green]{heading.text}[/bold green]")
        
        # この見出しの後に続くコンテンツを表示（次の見出しまで）
        content = []
        element = heading.next_sibling
        while element and not element.name in ['h2', 'h3']:
            if element.name == 'p':
                content.append(element.text)
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    content.append(f"• {li.text}")
            element = element.next_sibling if hasattr(element, 'next_sibling') else None
        
        if content:
            # 長いコンテンツは省略表示
            display_content = content[:2]
            if len(content) > 3:
                display_content.append("...")
                display_content.append(content[-1])
            for p in display_content:
                console.print(f"{indent}  {p}")

def display_seo_analysis(seo_analysis: SEOAnalysis) -> None:
    """SEO分析結果を表示する"""
    console.print(Panel.fit("[bold cyan]SEO分析結果[/bold cyan]", title="SEO Analysis"))
    
    console.print(f"[bold]読みやすさスコア:[/bold] {seo_analysis.readability_score}/100")
    
    console.print("\n[bold]キーワード密度:[/bold]")
    for keyword, density in seo_analysis.keyword_density.items():
        color = "green" if 0.5 <= density <= 3.0 else "red"
        console.print(f"  {keyword}: [bold {color}]{density}%[/bold {color}]")
    
    console.print("\n[bold red]問題点:[/bold red]")
    for issue in seo_analysis.issues:
        console.print(f"  • {issue}")
    
    console.print("\n[bold green]改善提案:[/bold green]")
    for rec in seo_analysis.recommendations:
        console.print(f"  • {rec}")

def save_article_to_file(article_html: str, filename: str = "generated_article.html") -> str:
    """記事をファイルに保存する"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(article_html)
        return os.path.abspath(filename)
    except Exception as e:
        console.print(f"[bold red]ファイル保存エラー: {e}[/bold red]")
        return ""

# --- オリジナル関数定義（ラッパー関数用） ---
# @function_toolデコレーターを使用すると元の関数は直接呼び出せなくなるため、ここに元の関数も保持しておく

async def _calculate_word_distribution(
    ctx: RunContextWrapper[ArticleGenerationContext],
    total_word_count: int,
    sections: List[SectionInfo]
) -> List[WordDistribution]:
    """
    総文字数を各セクションに分配します。
    
    Args:
        total_word_count: 総文字数
        sections: セクションのリスト
    
    Returns:
        List[WordDistribution]: 文字数が割り当てられたセクションのリスト
    """
    console.print(f"[dim]文字数分配計算中: 合計 {total_word_count} 文字, {len(sections)} セクション[/dim]")
    
    result = []
    
    # 重要度に基づいてセクションに文字数を分配
    # 導入部と結論は少し短く、メインコンテンツに多くの文字数を割り当てる
    
    # セクション数に基づく基本配分
    base_count_per_section = total_word_count / len(sections)
    
    for i, section in enumerate(sections):
        heading = section.heading
        description = section.description
        
        # 重要度係数の決定（導入と結論はやや短く）
        importance = 1.0  # デフォルト
        if i == 0 or "はじめに" in heading or "導入" in heading:
            importance = 0.8  # 導入部は少し短く
        elif i == len(sections) - 1 or "まとめ" in heading or "結論" in heading:
            importance = 0.7  # 結論部分も少し短く
        elif "主要" in heading or "重要" in heading or "核心" in heading:
            importance = 1.2  # 主要部分は少し長く
        
        # 文字数の計算
        word_count = int(base_count_per_section * importance)
        
        result.append(WordDistribution(
            heading=heading,
            description=description,
            target_word_count=word_count
        ))
    
    # 合計が目標文字数になるように調整
    current_total = sum(item.target_word_count for item in result)
    adjustment = total_word_count - current_total
    
    # 最も長いセクションに調整分を追加
    if adjustment != 0:
        max_section_idx = max(range(len(result)), key=lambda i: result[i].target_word_count)
        result[max_section_idx].target_word_count += adjustment
    
    return result

async def _format_article_to_html(
    ctx: RunContextWrapper[ArticleGenerationContext],
    title: str,
    sections: List[SectionForHTML],
    meta_description: str = ""
) -> str:
    """
    記事のセクションをHTML形式に整形します。
    
    Args:
        title: 記事のタイトル
        sections: セクションのリスト
        meta_description: メタディスクリプション
    
    Returns:
        str: 完全なHTML記事
    """
    console.print(f"[dim]HTML形式に整形中: {title}[/dim]")
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <meta name="description" content="{meta_description}">
</head>
<body>
    <article>
        <h1>{title}</h1>
"""
    
    for section in sections:
        html += f"""
        {section.content}
"""
    
    html += """
    </article>
</body>
</html>"""
    
    return html

async def _analyze_article_seo(
    ctx: RunContextWrapper[ArticleGenerationContext], 
    html_content: str,
    target_keywords: List[str]
) -> SEOAnalysis:
    """
    HTMLコンテンツのSEO分析を行います。
    
    Args:
        html_content: 分析するHTML形式の記事内容
        target_keywords: ターゲットキーワードのリスト
    
    Returns:
        SEOAnalysis: SEO分析結果
    """
    console.print("[dim]記事のSEO分析中...[/dim]")
    
    # HTMLからプレーンテキストを抽出
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    word_count = len(text_content)
    
    # キーワード密度の計算
    keyword_density = {}
    for keyword in target_keywords:
        count = text_content.lower().count(keyword.lower())
        density = count / (word_count / 100) if word_count > 0 else 0
        keyword_density[keyword] = round(density, 2)
    
    # 見出しの確認
    headings = [h.text for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
    heading_keywords = sum(1 for h in headings for k in target_keywords if k.lower() in h.lower())
    
    # 内部リンクのカウント
    internal_links = len(soup.find_all('a', href=True))
    
    # 画像の代替テキストチェック
    images = soup.find_all('img')
    images_with_alt = sum(1 for img in images if img.get('alt'))
    
    # パラグラフの長さチェック
    paragraphs = soup.find_all('p')
    avg_paragraph_length = sum(len(p.text) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    # 読みやすさスコアの計算 (簡易版)
    long_sentences = sum(1 for p in paragraphs for s in p.text.split('.') if len(s) > 100)
    readability_score = 100 - (long_sentences * 5) - (avg_paragraph_length > 400) * 20
    readability_score = max(0, min(100, readability_score))
    
    # 問題点の特定
    issues = []
    if heading_keywords < len(target_keywords) / 2:
        issues.append("見出しにキーワードが十分に含まれていません")
    
    if any(density > 5 for density in keyword_density.values()):
        issues.append("キーワードの過剰使用（キーワードスタッフィング）の可能性があります")
    
    if any(density < 0.5 for density in keyword_density.values()):
        issues.append("主要キーワードの使用頻度が低すぎます")
    
    if internal_links < 2:
        issues.append("内部リンクが少なすぎます")
    
    if images and images_with_alt < len(images):
        issues.append("一部の画像にalt属性がありません")
    
    if avg_paragraph_length > 400:
        issues.append("段落が長すぎます。読みやすさのために分割を検討してください")
    
    # 改善提案
    recommendations = []
    if heading_keywords < len(target_keywords) / 2:
        recommendations.append("見出しにもっとキーワードを含めてください")
    
    if any(density > 5 for density in keyword_density.values()):
        recommendations.append("キーワードの使用を自然にし、過剰使用を避けてください")
    
    if any(density < 0.5 for density in keyword_density.values()):
        recommendations.append("コンテンツ内でターゲットキーワードをもっと自然に使用してください")
    
    if internal_links < 2:
        recommendations.append("関連コンテンツへの内部リンクを追加してください")
    
    if images and images_with_alt < len(images):
        recommendations.append("すべての画像に適切なalt属性を追加してください")
    
    if avg_paragraph_length > 400:
        recommendations.append("長い段落を分割し、読みやすくしてください")
    
    return SEOAnalysis(
        keyword_density=keyword_density,
        readability_score=round(readability_score, 1),
        issues=issues,
        recommendations=recommendations
    )

# --- メイン処理関数 ---
async def generate_topic_proposals(
    agents: Dict[str, Agent], 
    context: ArticleGenerationContext
) -> List[ArticleTopic]:
    """リサーチエージェントを使ってトピック案を生成する"""
    console.print("[dim]記事トピックの提案を生成中...[/dim]")
    
    # リサーチエージェントを使用
    result = await Runner.run(
        starting_agent=agents["research_agent"],
        input="企業情報に基づいて、SEO効果の高い記事トピックを5つ提案してください。各トピックには、タイトル、説明、キーワード、提案理由を含めてください。",
        context=context,
        run_config=RunConfig(
            workflow_name="SEO記事-トピック提案",
        )
    )
    
    # モック実装（実際には、リサーチエージェントが返す結果を処理）
    topic_proposals = [
        ArticleTopic(
            title=f"{context.company_info.industry}における最新トレンド2025",
            description=f"{context.company_info.industry}の最新動向と今後の展望について解説する記事",
            keywords=[f"{context.company_info.industry} トレンド", "最新動向", "2025 予測", "業界分析"],
            rationale="トレンド記事は常に検索需要が高く、業界の専門家としての権威性を示せます。"
        ),
        ArticleTopic(
            title=f"{context.company_info.industry}初心者が知っておくべき10のこと",
            description=f"{context.company_info.industry}を始めたばかりの人向けの基本ガイド",
            keywords=[f"{context.company_info.industry} 初心者", "基礎知識", "ガイド", "入門"],
            rationale="初心者向けコンテンツは検索ボリュームが大きく、新規顧客獲得に効果的です。"
        ),
        ArticleTopic(
            title=f"{context.company_info.target_audience}のための{context.company_info.industry}活用術",
            description=f"{context.company_info.target_audience}が{context.company_info.industry}を最大限活用するための実践的アドバイス",
            keywords=[f"{context.company_info.target_audience}", "活用法", "実践テクニック", "効率化"],
            rationale="ターゲットオーディエンスに直接訴求するコンテンツは、コンバージョン率向上に効果的です。"
        ),
        ArticleTopic(
            title=f"{context.company_info.industry}の成功事例から学ぶ重要ポイント",
            description="業界の成功事例を分析し、そこから得られる教訓を紹介する記事",
            keywords=["成功事例", "ケーススタディ", "業界分析", "ベストプラクティス"],
            rationale="成功事例は読者の関心を引きやすく、実践的な価値を提供できます。"
        ),
        ArticleTopic(
            title=f"{context.company_info.industry}における一般的な誤解とその真実",
            description=f"{context.company_info.industry}に関する誤解を解き、正しい情報を提供する記事",
            keywords=["誤解", "真実", "専門知識", "正しい情報"],
            rationale="誤解を解くコンテンツは、専門知識をアピールしつつ、情報の正確性で信頼を構築できます。"
        )
    ]
    
    return topic_proposals

async def generate_structure_proposals(
    agents: Dict[str, Agent], 
    context: ArticleGenerationContext
) -> List[ArticleStructure]:
    """プランニングエージェントを使って記事構造案を生成する"""
    console.print("[dim]記事構造の提案を生成中...[/dim]")
    
    # プランニングエージェントを使用
    result = await Runner.run(
        starting_agent=agents["planning_agent"],
        input=f"選択されたトピック「{context.selected_topic.title}」に基づいて、効果的な記事構造を3つ提案してください。各構造には、タイトル、セクション構成、キーワード配置、文体・トーン、目標文字数を含めてください。",
        context=context,
        run_config=RunConfig(
            workflow_name="SEO記事-構造提案",
        )
    )
    
    # モック実装（実際には、プランニングエージェントが返す結果を処理）
    topic = context.selected_topic
    structure_proposals = [
        ArticleStructure(
            title=topic.title,
            sections=[
                SectionInfo(heading="はじめに", description=f"{topic.title}の概要と重要性について説明"),
                SectionInfo(heading=f"{topic.title}の基本", description="基本的な概念と知識を解説"),
                SectionInfo(heading="主要ポイント1", description="重要なポイントの詳細解説"),
                SectionInfo(heading="主要ポイント2", description="別の重要ポイントの詳細解説"),
                SectionInfo(heading="主要ポイント3", description="もう一つの重要ポイントの詳細解説"),
                SectionInfo(heading="実践的なアドバイス", description="読者が実際に活用できるヒントや方法"),
                SectionInfo(heading="よくある質問", description=f"{topic.title}に関するFAQ"),
                SectionInfo(heading="まとめ", description="記事の要点のまとめと次のステップ")
            ],
            tone="プロフェッショナルで明確な文体",
            target_word_count=3000,
            target_audience=context.company_info.target_audience,
            keywords=topic.keywords
        ),
        ArticleStructure(
            title=f"{topic.title} - 完全ガイド",
            sections=[
                SectionInfo(heading="導入", description=f"{topic.title}の背景と重要性"),
                SectionInfo(heading="現状分析", description=f"{topic.title}に関する現在の状況と課題"),
                SectionInfo(heading="ケーススタディ1", description="実際の成功事例とその分析"),
                SectionInfo(heading="ケーススタディ2", description="別の成功事例とその分析"),
                SectionInfo(heading="実践ステップ", description="読者が実行できる具体的な手順"),
                SectionInfo(heading="今後の展望", description=f"{topic.title}の将来動向と予測"),
                SectionInfo(heading="まとめと行動計画", description="記事の要約と読者が取るべき次のアクション")
            ],
            tone="教育的かつ実践的な文体",
            target_word_count=2500,
            target_audience=context.company_info.target_audience,
            keywords=topic.keywords
        ),
        ArticleStructure(
            title=f"{topic.title}を成功させる秘訣",
            sections=[
                SectionInfo(heading=f"なぜ今{topic.title}が重要なのか", description="トピックの時事的重要性"),
                SectionInfo(heading="よくある課題と解決策", description="一般的な問題点とその対処法"),
                SectionInfo(heading="専門家のアドバイス", description="業界専門家からの見解と助言"),
                SectionInfo(heading="ステップバイステップガイド", description="段階的な実践手順"),
                SectionInfo(heading="成功するためのツールと資源", description="役立つリソースとツールの紹介"),
                SectionInfo(heading="効果測定の方法", description="成果を測定・評価する方法"),
                SectionInfo(heading="結論と次のステップ", description="まとめと今後の行動プラン")
            ],
            tone="親しみやすく実用的な文体",
            target_word_count=2000,
            target_audience=context.company_info.target_audience,
            keywords=topic.keywords
        )
    ]
    
    return structure_proposals

async def generate_article_section(
    agents: Dict[str, Agent], 
    context: ArticleGenerationContext,
    section_info: WordDistribution
) -> Optional[ArticleSection]:
    """執筆エージェントを使って記事セクションを生成する"""
    if not context.selected_structure:
        return None
    
    heading = section_info.heading
    description = section_info.description
    target_word_count = section_info.target_word_count
    
    console.print(f"[dim]セクション「{heading}」を生成中...[/dim]")
    
    # 執筆エージェントを使用
    result = await Runner.run(
        starting_agent=agents["writing_agent"],
        input=f"記事「{context.selected_structure.title}」のセクション「{heading}」({description})を{target_word_count}文字程度でHTML形式で生成してください。自己完結的なHTMLを生成し、他のマークアップなどは含めないでください。キーワード {', '.join(context.selected_structure.keywords)} を自然に含めてください。文体・トーンは {context.selected_structure.tone} でお願いします。",
        context=context,
        run_config=RunConfig(
            workflow_name=f"SEO記事-セクション生成-{heading}",
        )
    )
    
    # 結果がHTML形式かどうかチェック
    content = result.final_output
    if not (content.startswith("<") and content.endswith(">")):
        # HTMLタグで囲む
        content = f"<section>\n<h2>{heading}</h2>\n<p>{content}</p>\n</section>"
    
    # 文字数をカウント（HTMLタグを除く）
    soup = BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text()
    word_count = len(text_content)
    
    return ArticleSection(
        heading=heading,
        content=content,
        word_count=word_count
    )

async def edit_and_optimize_article(
    agents: Dict[str, Agent], 
    context: ArticleGenerationContext
) -> Tuple[str, SEOAnalysis]:
    """編集エージェントを使って記事を編集・最適化する"""
    console.print("[dim]記事の編集と最適化中...[/dim]")
    
    current_article = context.get_current_article_html()
    
    # 編集エージェントを使用
    result = await Runner.run(
        starting_agent=agents["editing_agent"],
        input=f"生成された記事を編集し、SEO最適化してください。キーワード {', '.join(context.selected_structure.keywords if context.selected_structure else [])} が適切に使われているか確認してください。読みやすさ、論理構成、文法、表現を改善し、最終的なHTML形式の記事を提供してください。",
        context=context,
        run_config=RunConfig(
            workflow_name="SEO記事-編集最適化",
        )
    )
    
    # SEO分析 - デコレートされたfunctionは直接呼び出せないので、内部関数を呼び出す
    seo_analysis = await _analyze_article_seo(
        RunContextWrapper(context),
        current_article,
        context.selected_structure.keywords if context.selected_structure else []
    )
    
    # 最終的なHTML
    final_html = result.final_output
    if not isinstance(final_html, str):
        # 何らかの理由で文字列が返されなかった場合
        final_html = current_article
    
    # メタディスクリプションの生成（実際には編集エージェントが行う）
    meta_description = f"{context.selected_topic.title if context.selected_topic else '記事'} - {context.company_info.name if context.company_info else ''}による完全ガイド"
    
    # 完全なHTML形式に整形
    sections_for_html = []
    for section in context.sections:
        sections_for_html.append(SectionForHTML(
            heading=section.heading,
            content=section.content
        ))
    
    # デコレートされたfunctionは直接呼び出せないので、内部関数を呼び出す
    formatted_html = await _format_article_to_html(
        RunContextWrapper(context),
        context.selected_structure.title if context.selected_structure else "記事タイトル",
        sections_for_html,
        meta_description
    )
    
    return formatted_html, seo_analysis

# --- メイン実行関数 ---
async def run_article_generation_workflow():
    """記事生成ワークフローを実行する"""
    # エージェントの作成
    agents = create_agents()
    
    # コンテキストの初期化
    context = ArticleGenerationContext()
    
    # ステップ1: 企業情報の収集
    company_info = get_company_info()
    context.company_info = company_info
    context.current_state = "企業情報収集完了"
    
    # ステップ2: トピック提案の生成
    with Progress() as progress:
        task = progress.add_task("[cyan]トピック提案を生成中...", total=1)
        topics = await generate_topic_proposals(agents, context)
        progress.update(task, completed=1)
    
    context.topics = topics
    topic_idx = display_topic_proposals(topics)
    context.selected_topic = topics[topic_idx]
    context.current_state = "トピック選択完了"
    
    # ステップ3: 記事構造の提案
    with Progress() as progress:
        task = progress.add_task("[cyan]記事構造を設計中...", total=1)
        structures = await generate_structure_proposals(agents, context)
        progress.update(task, completed=1)
    
    context.article_structures = structures
    structure_idx = display_structure_proposals(structures)
    context.selected_structure = structures[structure_idx]
    context.current_state = "記事構造確定"
    
    # ステップ4: セクションごとの文字数分配
    with Progress() as progress:
        task = progress.add_task("[cyan]文字数分配を計算中...", total=1)
        # 直接デコレート済み関数は呼び出せないので、内部関数を使用
        distributed_sections = await _calculate_word_distribution(
            RunContextWrapper(context),
            context.selected_structure.target_word_count,
            context.selected_structure.sections
        )
        progress.update(task, completed=1)
    
    # ステップ5: セクションごとの生成
    total_sections = len(distributed_sections)
    with Progress() as progress:
        task = progress.add_task("[cyan]記事を生成中...", total=total_sections)
        
        for i, section_info in enumerate(distributed_sections):
            section = await generate_article_section(agents, context, section_info)
            if section:
                context.sections.append(section)
                progress.update(task, completed=i+1)
    
    context.current_state = "記事生成完了"
    
    # ステップ6: 記事の編集・最適化
    with Progress() as progress:
        task = progress.add_task("[cyan]記事の編集と最適化中...", total=1)
        final_html, seo_analysis = await edit_and_optimize_article(agents, context)
        progress.update(task, completed=1)
    
    context.final_article_html = final_html
    context.seo_analysis = seo_analysis
    context.current_state = "記事編集完了"
    
    # ステップ7: 記事プレビューと保存
    display_article_preview(final_html)
    display_seo_analysis(seo_analysis)
    
    # 記事の保存
    if Confirm.ask("[bold]生成された記事をファイルに保存しますか？[/bold]", default=True):
        filename = Prompt.ask("[bold]保存するファイル名[/bold]", default="generated_article.html")
        saved_path = save_article_to_file(final_html, filename)
        if saved_path:
            console.print(f"[bold green]記事を保存しました: {saved_path}[/bold green]")
    
    console.print(Panel.fit("[bold green]記事生成完了！[/bold green]", title="完了"))

# --- メインエントリーポイント ---
if __name__ == "__main__":
    try:
        asyncio.run(run_article_generation_workflow())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]プログラムが中断されました。[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]エラーが発生しました: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())