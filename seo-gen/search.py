import asyncio
from agents import Agent, Runner, WebSearchTool

async def main():
    agent = Agent(
        name="Web searcher",
        instructions="You are a helpful agent.",
        tools=[WebSearchTool(user_location={"type": "approximate", "country": "JP"})],
    )
    result = await Runner.run(
        agent,
        "search the web for 'local sports news' and give me 1 interesting update in a sentence."
    )
    print(result.final_output)

asyncio.run(main())
