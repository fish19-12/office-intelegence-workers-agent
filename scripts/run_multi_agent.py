"""Example runner to demonstrate the multi-agent system."""
import asyncio
import logging

from agent_message_bus import MessageBus
from supervisor_agent import SupervisorAgent
from specialist_agents import DataAgent, ReportAgent, CommunicationAgent, RiskAgent, SearchAgent

# A very small mock LLM that returns simple JSON for planner prompts
class MockLLM:
    def generate(self, prompt: str) -> str:
        # Very naive: if 'break' in prompt, return a sample plan
        if 'Return ONLY a JSON array of tasks' in prompt:
            return '[{"agent": "DataAgent", "task_id": "t1", "payload": {"sheet_url": "http://example.com/sheet"}}, {"agent": "ReportAgent", "task_id": "t2", "payload": {"report_name": "out.docx", "sections": []}}, {"agent": "CommunicationAgent", "task_id": "t3", "payload": {"channel": "email", "message": "Report ready", "recipients": ["user@example.com"]}}]'
        return '[]'


async def main():
    logging.basicConfig(level=logging.INFO)
    bus = MessageBus()
    llm = MockLLM()

    # mcp is optional for this demo; pass None
    mcp = None

    # Define specialists and their allowed tools (optional)
    specialists = {
        "DataAgent": {"allowed_tools": ["google_sheets_read"]},
        "ReportAgent": {"allowed_tools": []},
        "CommunicationAgent": {"allowed_tools": ["email_send"]},
        "RiskAgent": {"allowed_tools": []},
        "SearchAgent": {"allowed_tools": ["document_search"]},
    }

    # Instantiate agents
    data = DataAgent("DataAgent", llm, mcp, bus, allowed_tools=specialists["DataAgent"]["allowed_tools"])
    report = ReportAgent("ReportAgent", llm, mcp, bus, allowed_tools=specialists["ReportAgent"]["allowed_tools"])
    comm = CommunicationAgent("CommunicationAgent", llm, mcp, bus, allowed_tools=specialists["CommunicationAgent"]["allowed_tools"])
    risk = RiskAgent("RiskAgent", llm, mcp, bus, allowed_tools=specialists["RiskAgent"]["allowed_tools"])
    search = SearchAgent("SearchAgent", llm, mcp, bus, allowed_tools=specialists["SearchAgent"]["allowed_tools"])

    supervisor = SupervisorAgent("Supervisor", llm, mcp, bus, specialists)

    # Start agents
    await data.start()
    await report.start()
    await comm.start()
    await risk.start()
    await search.start()
    await supervisor.start()

    # Dispatch a sample user goal
    result = await supervisor.dispatch_user_goal("Generate a loan report and notify the user by email")
    print("Dispatch result:\n", result)

    # Allow a short grace period then stop
    await asyncio.sleep(1)

    await data.stop()
    await report.stop()
    await comm.stop()
    await risk.stop()
    await search.stop()
    await supervisor.stop()

if __name__ == "__main__":
    asyncio.run(main())
