"""
WebSearchResearchAgent - OpenAI Agents SDKを使用したウェブ検索リサーチエージェント

このスクリプトはOpenAI Agents SDKを使用して、指定されたトピックに対して
深いリサーチを行うエージェントを実装します。WebSearchツールを活用して
複数の情報源からデータを収集し、構造化されたレポートを生成します。

使用方法:
スクリプトを実行: python research_agent.py <トピック> [--queries <クエリ数>]
   例: python research_agent.py "人工知能の歴史" --queries 5
"""

import os
import sys
import asyncio
import json
import argparse
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field

# OpenAI Agent SDKのインポート
from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    trace,
    RunContextWrapper,
    handoff,
    set_tracing_disabled,
)

# APIキーが設定されているか確認
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("エラー: OPENAI_API_KEYが設定されていません。")
    print("環境変数としてAPIキーを設定してください。例: export OPENAI_API_KEY='your-key'")
    sys.exit(1)

# デバッグを有効化
DEBUG = True

# トレーシングの無効化オプション（必要に応じて有効化）
# set_tracing_disabled(True)

# データモデルの定義
class SearchQuery(BaseModel):
    """検索クエリの構造化モデル"""
    query: str = Field(description="検索する具体的なクエリ")
    purpose: str = Field(description="このクエリで何を明らかにしたいのか")


class ResearchPlan(BaseModel):
    """リサーチ計画の構造化モデル"""
    topic: str = Field(description="リサーチするトピック")
    search_queries: List[SearchQuery] = Field(description="トピックを探求するための検索クエリリスト")
    focus_areas: List[str] = Field(description="特に注目すべき領域や側面")


class FactInfo(BaseModel):
    """収集した重要な事実の構造化モデル"""
    fact: str = Field(description="重要な事実や発見")
    source: Optional[str] = Field(description="情報の出所（ウェブサイトURLなど）", default=None)
    relevance: str = Field(description="この事実がリサーチトピックにどう関連するか")


class ResearchSummary(BaseModel):
    """検索結果の要約モデル"""
    query: str = Field(description="実行された検索クエリ")
    key_findings: List[FactInfo] = Field(description="この検索から発見された主要な事実")
    follow_up_queries: Optional[List[str]] = Field(description="さらなる探索のための追加クエリ提案", default=None)


class ReportSection(BaseModel):
    """レポートセクションの構造化モデル"""
    title: str = Field(description="セクションのタイトル")
    content: str = Field(description="セクションの内容")


class ResearchReport(BaseModel):
    """最終的なリサーチレポートの構造化モデル"""
    title: str = Field(description="レポートのタイトル")
    summary: str = Field(description="リサーチの要約（200-300文字）")
    key_findings: List[str] = Field(description="主要な発見や洞察のリスト")
    sections: List[ReportSection] = Field(description="レポートのセクション（タイトルと内容）")
    sources: List[str] = Field(description="情報ソースのリスト")
    follow_up_topics: Optional[List[str]] = Field(description="さらなる探求のための提案トピック", default=None)


# コンテキストの定義
@dataclass
class ResearchContext:
    """エージェント間で共有されるコンテキスト"""
    topic: str
    query_count: int = 5  # デフォルトのクエリ数
    research_plan: Optional[ResearchPlan] = None
    collected_facts: List[FactInfo] = field(default_factory=list)
    search_summaries: List[ResearchSummary] = field(default_factory=list)
    current_stage: str = "planning"  # planning, researching, writing
    current_query_index: int = 0
    debug_log: List[str] = field(default_factory=list)
    
    def add_debug(self, message: str):
        """デバッグメッセージを追加"""
        if DEBUG:
            print(f"[DEBUG] {message}")
        self.debug_log.append(message)

    def update_plan(self, plan: ResearchPlan):
        """リサーチプランを更新し、ステージを進める"""
        self.research_plan = plan
        
        # クエリ数が指定されている場合、クエリリストを制限
        if self.query_count > 0 and len(plan.search_queries) > self.query_count:
            self.add_debug(f"クエリ数を{len(plan.search_queries)}から{self.query_count}に制限します")
            limited_queries = plan.search_queries[:self.query_count]
            self.research_plan.search_queries = limited_queries
        
        self.current_stage = "researching"
        self.add_debug(f"リサーチプラン設定完了: {self.research_plan.topic}、検索クエリ数: {len(self.research_plan.search_queries)}")

    def add_summary(self, summary: ResearchSummary):
        """検索要約を追加し、次のクエリに進む"""
        self.search_summaries.append(summary)
        self.current_query_index += 1
        self.add_debug(f"検索要約を追加: クエリ '{summary.query}'、発見数: {len(summary.key_findings)}")
        
        # 全クエリの検索が完了した場合、ライティングステージへ
        if self.research_plan and self.current_query_index >= len(self.research_plan.search_queries):
            self.current_stage = "writing"
            self.add_debug("全検索クエリ完了。ライティングステージへ移行")


# カスタムツールの定義
@function_tool
def save_important_fact(ctx: RunContextWrapper[ResearchContext], fact: str, source: str = None, relevance: str = None) -> str:
    """リサーチ中に発見した重要な事実を保存する"""
    new_fact = FactInfo(
        fact=fact,
        source=source,
        relevance=relevance or "関連情報"
    )
    ctx.context.collected_facts.append(new_fact)
    ctx.context.add_debug(f"事実を保存: {fact[:50]}...")
    return f"事実を保存しました: {fact}"


@function_tool
def get_research_status(ctx: RunContextWrapper[ResearchContext]) -> str:
    """現在のリサーチステータスを返す"""
    context = ctx.context
    status = {
        "topic": context.topic,
        "current_stage": context.current_stage,
        "has_research_plan": context.research_plan is not None,
        "collected_facts_count": len(context.collected_facts),
        "search_summaries_count": len(context.search_summaries),
    }
    
    if context.research_plan:
        status["total_queries"] = len(context.research_plan.search_queries)
        status["current_query_index"] = context.current_query_index
        
        if context.current_stage == "researching" and context.current_query_index < len(context.research_plan.search_queries):
            status["current_query"] = context.research_plan.search_queries[context.current_query_index].query
    
    ctx.context.add_debug(f"ステータス取得: {status}")
    return json.dumps(status, ensure_ascii=False, indent=2)


# エージェントの定義
def create_planning_agent() -> Agent[ResearchContext]:
    """トピックを分析して研究計画を作成するエージェント"""
    return Agent[ResearchContext](
        name="Planning Agent",
        instructions="""
        あなたはリサーチプランナーです。与えられたトピックを分析し、
        体系的なリサーチ計画を作成します。
        
        あなたのタスク:
        1. トピックを分析し、理解します
        2. トピックを探索するための検索クエリを作成します
           - 通常は5-8個のクエリを作成しますが、システムの制限に応じて調整されます
           - 各クエリは具体的かつ多様な情報を収集できるものであること
        3. リサーチで特に注目すべき3-5の重要な領域や側面を特定します
        4. これらの情報から構造化されたResearchPlanを作成します
        
        検索クエリは具体的で、多様な情報を収集できるように設計してください。
        各クエリには明確な目的を付けてください。
        
        注意: このプランは後続のResearcher Agentによって使用されます。正確かつ詳細なプランを作成してください。
        ResearchPlan形式で回答してください。出力はJSONとして正しく解析可能なものにしてください。
        """,
        model="gpt-4.1-mini",
        output_type=ResearchPlan,
        tools=[get_research_status],
    )


def create_researcher_agent(search_location=None) -> Agent[ResearchContext]:
    """ウェブ検索を実行して情報を収集するエージェント"""
    # 検索ロケーションが指定されていれば使用、なければデフォルト
    search_tool = WebSearchTool(user_location=search_location) if search_location else WebSearchTool()
    
    return Agent[ResearchContext](
        name="Researcher Agent",
        instructions="""
        あなたはリサーチャーです。指定された検索クエリに基づいてウェブ検索を実行し、
        関連する重要な情報を収集します。
        
        あなたのタスク:
        1. get_research_statusツールを使用して現在のリサーチステータスを確認します
        2. 現在の検索クエリを使用してウェブ検索を実行します
        3. 検索結果から重要かつ関連性の高い事実を特定します
        4. 各事実について、情報源と関連性を記録します
        5. save_important_fact ツールを使って見つけた重要な事実を保存します
        6. 検索結果の要約をResearchSummaryとして作成します
        
        事実を報告する際は:
        - 正確さを重視してください
        - 情報源を常に記録してください
        - 矛盾する情報がある場合はそれを明示してください
        - 確かな事実と推測を区別してください
        
        注意: 一度に1つの検索クエリのみを処理します。ResearchSummary形式で回答し、
        次の検索クエリはシステムによって自動的に割り当てられます。
        """,
        model="gpt-4.1-mini",
        tools=[search_tool, save_important_fact, get_research_status],
        output_type=ResearchSummary,
    )


def create_writer_agent() -> Agent[ResearchContext]:
    """収集した情報から最終レポートを作成するエージェント"""
    return Agent[ResearchContext](
        name="Writer Agent",
        instructions="""
        あなたはライターです。収集された事実や要約から包括的なリサーチレポートを作成します。
        
        あなたのタスク:
        1. get_research_statusツールを使用してコンテキスト情報を確認します
        2. コンテキスト内のすべての収集事実と検索要約を分析します
        3. 情報を整理し、論理的な構造でレポートを組み立てます
        4. 以下を含むResearchReportを作成します：
           - 明確なタイトル
           - 簡潔な要約（200-300文字）
           - 主要な発見のリスト
           - 整理されたセクション（各セクションにはtitleとcontent属性があります）
           - 使用した情報源のリスト
           - さらなる探求のための提案トピック
        
        レポート作成の際は:
        - 客観的な立場を維持してください
        - すべての主張には裏付けとなる収集事実を使用してください
        - 複雑な概念は明確に説明してください
        - 事実間の関連性や対比を示してください
        
        注意: これが最終成果物です。詳細で包括的なレポートを作成してください。
        sectionsフィールドの各要素は、titleとcontentの2つの属性を持つオブジェクトです。
        titleはセクションの見出し、contentはセクションの本文です。
        ResearchReport形式で回答してください。
        """,
        model="gpt-4.1-mini",
        tools=[get_research_status],
        output_type=ResearchReport,
    )


def create_triage_agent() -> Agent[ResearchContext]:
    """リサーチワークフローを調整する中央エージェント"""
    planning_agent = create_planning_agent()
    researcher_agent = create_researcher_agent()
    writer_agent = create_writer_agent()
    
    return Agent[ResearchContext](
        name="Research Coordinator",
        instructions="""
        あなたはリサーチコーディネーターです。リサーチワークフロー全体を管理・調整します。

        コンテキストの「current_stage」変数に基づいて以下のように動作します:
        
        1. current_stage = "planning" の場合:
           - Planning Agentにハンドオフして、リサーチプランを作成させます
           - プランが作成されると自動的に current_stage = "researching" に更新されます
        
        2. current_stage = "researching" の場合:
           - Researcher Agentにハンドオフして、現在の検索クエリについて調査させます
           - 各クエリの調査が完了すると current_query_index が増加します
           - すべてのクエリが完了すると自動的に current_stage = "writing" に更新されます
        
        3. current_stage = "writing" の場合:
           - Writer Agentにハンドオフして、収集した情報から最終レポートを作成させます
           - レポートが完成したら、ユーザーに提示します
        
        get_research_status ツールを使用して、現在の進行状況を確認し、
        適切なエージェントにハンドオフしてください。各ステージが確実に
        完了してから次のステージに進むようにしてください。

        重要: 各エージェントからの出力がコンテキストに適切に保存されることを確認してください。
        Planning Agentの出力はresearch_planに、Researcher Agentの出力はsearch_summariesに、
        Writer Agentの出力は最終レポートとして返されます。
        """,
        model="gpt-4.1-mini",
        tools=[get_research_status],
        handoffs=[
            handoff(planning_agent),
            handoff(researcher_agent),
            handoff(writer_agent),
        ],
    )


# コンテキストを更新するための関数
async def update_context_from_result(result, context: ResearchContext):
    """エージェントの実行結果からコンテキストを更新する"""
    if hasattr(result, "final_output") and result.final_output:
        if isinstance(result.final_output, ResearchPlan):
            context.update_plan(result.final_output)
            return True
        elif isinstance(result.final_output, ResearchSummary):
            context.add_summary(result.final_output)
            return True
        elif isinstance(result.final_output, ResearchReport):
            # 最終レポートはフローの最後なので、コンテキストへの特別な更新は不要
            context.add_debug("最終レポート生成完了")
            return True
    
    context.add_debug("エージェントの結果からコンテキスト更新に失敗")
    return False


async def run_research(topic: str, query_count: int = 0) -> Union[ResearchReport, str]:
    """指定されたトピックに関するリサーチを実行し、レポートを生成する"""
    # コンテキストの初期化
    context = ResearchContext(topic=topic, query_count=query_count)
    context.add_debug(f"リサーチ開始: '{topic}', クエリ数上限: {query_count if query_count > 0 else '無制限'}")
    
    # トリアージエージェントの作成
    triage_agent = create_triage_agent()
    
    # トレーシングの開始（オプション）
    with trace(workflow_name=f"Research on {topic}"):
        # 最初のエージェント実行
        result = await Runner.run(
            triage_agent,
            input=f"次のトピックについて徹底的なリサーチを行い、包括的なレポートを作成してください: {topic}",
            context=context,
            max_turns=50,  # 最大ターン数を増やす
        )
        
        # エージェント結果を処理
        if hasattr(result, "final_output") and result.final_output:
            # ResearchReportオブジェクトの場合は成功
            if isinstance(result.final_output, ResearchReport):
                return result.final_output
            else:
                # 最終出力が期待と異なる場合は、現在の状態を返す
                context.add_debug(f"予期しない最終出力タイプ: {type(result.final_output).__name__}")
                if context.current_stage == "planning" and isinstance(result.final_output, ResearchPlan):
                    # Planning Agentの出力を手動でコンテキストに設定
                    context.update_plan(result.final_output)
                    
                    # Researcher Agentを手動で実行
                    context.add_debug("Researcher Agentを手動で実行")
                    researcher_agent = create_researcher_agent()
                    
                    current_queries = context.research_plan.search_queries
                    for i, query_obj in enumerate(current_queries):
                        context.current_query_index = i
                        query = query_obj.query
                        
                        context.add_debug(f"検索クエリ実行中 ({i+1}/{len(current_queries)}): {query}")
                        result = await Runner.run(
                            researcher_agent,
                            input=f"次の検索クエリを実行してください: {query}",
                            context=context,
                            max_turns=15,
                        )
                        
                        if hasattr(result, "final_output") and isinstance(result.final_output, ResearchSummary):
                            context.add_summary(result.final_output)
                        else:
                            context.add_debug(f"Researcher Agent結果エラー: {type(result.final_output).__name__ if hasattr(result, 'final_output') else 'No output'}")
                    
                    # Writer Agentを手動で実行
                    context.current_stage = "writing"
                    context.add_debug("Writer Agentを手動で実行")
                    writer_agent = create_writer_agent()
                    
                    result = await Runner.run(
                        writer_agent,
                        input="収集した情報から包括的なリサーチレポートを作成してください。",
                        context=context,
                        max_turns=15,
                    )
                    
                    if hasattr(result, "final_output") and isinstance(result.final_output, ResearchReport):
                        return result.final_output
                
                # フローが途中で止まった場合のフォールバック
                return f"リサーチが完了しませんでした。現在のステージ: {context.current_stage}\nデバッグログ: {context.debug_log}"
        else:
            context.add_debug("エージェントからの結果がありません")
            return "エージェントから結果を得られませんでした。デバッグログ: " + str(context.debug_log)


def format_report(report: ResearchReport) -> str:
    """レポートをフォーマットして表示用の文字列を返す"""
    output = "\n" + "="*80 + "\n"
    output += f"## {report.title}\n"
    output += "="*80 + "\n"
    
    output += "\n### 要約\n"
    output += report.summary + "\n"
    
    output += "\n### 主要な発見\n"
    for i, finding in enumerate(report.key_findings, 1):
        output += f"{i}. {finding}\n"
    
    output += "\n### 詳細セクション\n"
    for section in report.sections:
        output += f"\n#### {section.title}\n"
        output += section.content + "\n"
    
    output += "\n### 情報源\n"
    for i, source in enumerate(report.sources, 1):
        output += f"{i}. {source}\n"
    
    if report.follow_up_topics:
        output += "\n### さらなる探求トピック\n"
        for i, topic in enumerate(report.follow_up_topics, 1):
            output += f"{i}. {topic}\n"
    
    return output


# コマンドライン引数を解析する関数
def parse_args():
    parser = argparse.ArgumentParser(description="WebSearchResearchAgent - トピックに関するリサーチを実行するエージェント")
    parser.add_argument("topic", help="リサーチするトピック")
    parser.add_argument("--queries", "-q", type=int, default=0, help="検索クエリの数 (デフォルト: 無制限)")
    return parser.parse_args()


# メイン関数
async def main():
    # コマンドライン引数を解析
    args = parse_args()
    topic = args.topic
    query_count = args.queries
    
    query_count_info = f"クエリ数: {query_count}" if query_count > 0 else "クエリ数: 無制限(AI判断)"
    print(f"トピック '{topic}' についてリサーチを開始します... {query_count_info}")
    
    try:
        result = await run_research(topic, query_count)
        
        if isinstance(result, ResearchReport):
            # レポートをフォーマットして表示
            formatted_report = format_report(result)
            print(formatted_report)
        else:
            # エラーメッセージまたは中間結果
            print("\n結果:")
            print(result)
            
    except KeyboardInterrupt:
        print("\nリサーチが中断されました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


# Pythonスクリプトとして実行された場合に実行
if __name__ == "__main__":
    asyncio.run(main())