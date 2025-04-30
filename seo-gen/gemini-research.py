# -*- coding: utf-8 -*-
import os
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, BadRequestError
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

# --- Agents SDK ---
from agents import (
    Agent,
    Runner,
    RunConfig,
    WebSearchTool, # WebSearchTool をインポート
    ModelSettings,
    RunContextWrapper,
    # エラーハンドリング用
    AgentsException,
    MaxTurnsExceeded,
    ModelBehaviorError,
    UserError,
)

# コンソール出力用
console = Console()

# 環境変数の読み込み (例: .env ファイルから)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OpenAI APIキー (OPENAI_API_KEY) が設定されていません。.envファイルを確認してください。")

# --- グローバル設定 ---
async_client = AsyncOpenAI(api_key=API_KEY)
# 使用するモデルを指定 (例: gpt-4o-mini, gpt-4.1-mini など)
# WebSearchTool は o3 モデルでは動作しない可能性があるため、gpt-4系を推奨
MODEL = "gpt-4o-mini"

# --- Pydanticモデル定義 (エージェントの出力型) ---
class ResearchReport(BaseModel):
    """調査結果を格納するレポートモデル"""
    topic: str = Field(description="調査対象のトピック")
    summary: str = Field(description="調査結果の要約")
    key_findings: List[str] = Field(description="主要な発見事項のリスト")
    sources: Optional[List[str]] = Field(default=None, description="参照した情報源のURLリスト (WebSearchToolが提供する場合)")

# --- コンテキストクラス (今回は未使用だが、拡張用に定義) ---
class ResearchContext:
    """エージェントやツールが共有するコンテキスト (今回はシンプルにするため未使用)"""
    pass # 必要に応じてフィールドを追加

# --- エージェント定義 ---
RESEARCH_AGENT_INSTRUCTIONS = """
あなたは優秀なリサーチアシスタントです。
ユーザーから指定されたトピックについて、WebSearchTool を使用して徹底的に調査してください。

**あなたのタスク:**
1. ユーザーの入力から調査すべきトピックを特定します。
2. `WebSearchTool` を使用して、関連性の高い最新情報を収集します。必要に応じて複数回検索を実行してください。
3. 収集した情報に基づいて、以下の要素を含む詳細なレポートを作成します。
    - topic: 調査したトピック
    - summary: 調査結果全体の簡潔な要約
    - key_findings: 最も重要と思われる発見事項を箇条書きリストで複数提示
    - sources: (可能であれば) 参照した主要な情報源のURLリスト
4. 最終的な応答は、必ず `ResearchReport` 型のJSON形式で出力してください。
"""

research_agent = Agent[ResearchContext]( # コンテキスト型を指定 (今回は未使用)
    name="WebResearchAgent",
    instructions=RESEARCH_AGENT_INSTRUCTIONS,
    model=MODEL,
    model_settings=ModelSettings(temperature=0.5), # 応答の創造性を調整
    tools=[
        WebSearchTool(
            # search_context_size="high" # オプション: 検索コンテキストのサイズを指定 (low, medium, high)
            # user_location="Tokyo, Japan" # オプション: 検索地域を指定
        )
    ],
    output_type=ResearchReport, # Pydanticモデルを出力型として指定
)

# --- メイン実行部分 ---
async def main():
    console.print("[bold magenta]Web検索リサーチエージェントへようこそ！[/]")

    # 今回はコンテキストを使用しないためNoneを渡す
    research_context = None # ResearchContext() # 必要に応じて初期化

    run_config = RunConfig(
        workflow_name="WebResearchWorkflow",
        # max_turns=5 # オプション: 最大ターン数を設定
    )

    total_tokens_used = 0

    while True:
        console.rule()
        try:
            # ユーザーから調査トピックを入力
            topic_to_research = Prompt.ask("[bold yellow]調査したいトピックを入力してください (終了するには /quit)[/]", default="")

            if not topic_to_research.strip():
                continue
            if topic_to_research.lower() in ["/quit", "/exit", "終了", "exit", "quit"]:
                break

            console.print(f"[dim]'{topic_to_research}' について調査を開始します...[/dim]")

            # --- Agent実行 ---
            # input は文字列でトピックを渡す
            result = await Runner.run(
                starting_agent=research_agent,
                input=topic_to_research,
                context=research_context, # コンテキストを渡す (今回はNone)
                run_config=run_config,
                max_turns=5 # Runner.run に max_turns を渡す
            )

            # --- トークン数表示 ---
            run_tokens = 0
            if result.raw_responses:
                for resp in result.raw_responses:
                    usage = getattr(resp, 'usage', None)
                    if usage and hasattr(usage, 'total_tokens'):
                        run_tokens += usage.total_tokens
            if run_tokens > 0:
                total_tokens_used += run_tokens
                console.print(f"[cyan]今回トークン(推定): {run_tokens}, 累計: {total_tokens_used}[/]")

            # --- Agentの出力を取得・検証 ---
            final_output_raw = result.final_output
            agent_report: Optional[ResearchReport] = None

            if isinstance(final_output_raw, ResearchReport):
                agent_report = final_output_raw
            elif isinstance(final_output_raw, str):
                 # 文字列で返ってきた場合、JSONパースを試みる (フォールバック)
                try:
                    parsed = json.loads(final_output_raw)
                    agent_report = ResearchReport(**parsed)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    console.print("[yellow]警告: AIからの応答が期待したレポート形式ではありません。応答内容を表示します。[/yellow]")
                    console.print(f"生データ: {final_output_raw}")
            elif final_output_raw is None:
                 console.print("[yellow]AIからの応答がありませんでした。[/yellow]")
            else:
                 console.print(f"[red]エラー: AIから予期しない型の応答がありました: {type(final_output_raw)}[/red]")
                 console.print(f"生データ: {final_output_raw}")


            # --- レポート表示 ---
            if agent_report:
                console.print("\n[bold green]--- 調査レポート ---[/]")
                console.print(f"[bold]トピック:[/bold] {agent_report.topic}")
                console.print(f"\n[bold]要約:[/bold]\n{agent_report.summary}")
                console.print("\n[bold]主要な発見事項:[/bold]")
                for finding in agent_report.key_findings:
                    console.print(f"- {finding}")
                if agent_report.sources:
                    console.print("\n[bold]参照ソース:[/bold]")
                    for source in agent_report.sources:
                        console.print(f"- {source}")
                console.print("[bold green]--- レポート終 ---[/]")

        # --- ループ内のエラーハンドリング ---
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            console.print(f"[bold red]Agent実行エラー ({type(e).__name__}): {e}[/]")
            console.print("[yellow]別のトピックを試すか、設定を確認してください。[/yellow]")
        except BadRequestError as e:
            console.print(f"[bold red]OpenAI APIエラー: {e}[/]")
            console.print("[yellow]APIキーや接続を確認してください。[/yellow]")
        except AgentsException as e:
            console.print(f"[bold red]Agent SDK エラー: {e}[/]")
        except EOFError:
            console.print("\n[bold magenta]入力がキャンセルされました。終了します。[/]")
            break
        except KeyboardInterrupt:
            console.print("\n[bold magenta]中断しました。終了します。[/]")
            break
        except Exception as e:
            console.print(f"[bold red]予期せぬエラーが発生しました: {e}[/]")
            import traceback
            traceback.print_exc()

    console.print("[bold magenta]リサーチエージェントを終了します。[/]")

# --- プログラムのエントリポイント ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n[bold red]プログラム実行中に致命的なエラーが発生しました: {e}[/]")
        import traceback
        traceback.print_exc()
