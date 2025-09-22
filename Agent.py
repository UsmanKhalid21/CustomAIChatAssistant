import os
import sys
import re

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain.agents import create_tool_calling_agent, AgentExecutor

# === Import custom modules ===
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_file = os.path.join(current_dir, "Rag", "rag.py")
web_scrape_file = os.path.join(current_dir, "Web_Scraping", "test2.py")

sys.path.append(os.path.dirname(rag_file))
sys.path.append(os.path.dirname(web_scrape_file))

import rag
import test2

# === Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# ======================
#  TOOLS DEFINITIONS
# ======================

# ---- RAG Tool ----
class RAGInput(BaseModel):
    question: str | None = Field(None, description="Personal or project-related question")
    query: str | None = Field(None, description="Alias for question")

def rag_tool_func(question: str = None, query: str = None) -> str:
    q = question or query
    return rag.rag_query(q, memory.chat_memory.messages)

rag_tool = StructuredTool.from_function(
    func=rag_tool_func,
    name="RAG",
    description="Answer personal/project-related questions (e.g., Usman, Kahuta, portfolio, notes).",
    args_schema=RAGInput,
)

# ---- Web Scraping Tool ----
class WebScrapeInput(BaseModel):
    question: str | None = Field(None, description="General cricket/world sports question")
    query: str | None = Field(None, description="Alias for question")

def web_scrape_tool_func(question: str = None, query: str = None) -> str:
    q = question or query
    return test2.web_query(q, memory.chat_memory.messages)

web_scrape = StructuredTool.from_function(
    func=web_scrape_tool_func,
    name="WebScraping",
    description="Answer general/world cricket questions (batsmen, bowlers, teams, stats, ICC info).",
    args_schema=WebScrapeInput,
)

# ---- Calculator Tool ----
class CalculatorInput(BaseModel):
    expression: str | None = Field(None, description="Math expression like '2x2', '4+5', '10/2', '9-4'")
    query: str | None = Field(None, description="Alias for expression")

def calculate(expression: str = None, query: str = None) -> str:
    try:
        expr = (expression or query or "").replace(" ", "")
        numbers = re.findall(r"\d+(?:\.\d+)?", expr)
        operator = re.findall(r"[+\-*/x]", expr)

        if len(numbers) < 2 or not operator:
            return "Please provide a valid expression like '4+5', '10/2', '3*7', or '9-4'."

        num1, num2 = map(float, numbers[:2])
        op = operator[0]

        if op in ["x", "*"]:
            result = num1 * num2
        elif op == "+":
            result = num1 + num2
        elif op == "-":
            result = num1 - num2
        elif op == "/":
            if num2 == 0:
                return "Error: Division by zero is not allowed."
            result = num1 / num2
        else:
            return "Unsupported operation."

        return str(int(result)) if result.is_integer() else str(result)

    except Exception as e:
        return f"Error: {str(e)}"

calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="Calculator",
    description="Perform basic arithmetic operations like '4x5', '2*2', '10+3', '15/3'.",
    args_schema=CalculatorInput,
)

# ======================
#  COMBINING TOOLS
# ======================
tools = [rag_tool, web_scrape, calculator_tool]

System_prompt = """
You are an AI assistant with access to three tools.

Available tools:
- RAG: retrieves answers from Pinecone notes (personal and project info about Usman, Kahuta, portfolio, etc.).
- WebScraping: retrieves answers from https://www.icc-cricket.com/ (general cricket/sports info).
- Calculator: performs arithmetic (add, subtract, multiply, divide).

Rules:

1. Tool Selection
   - If the query mentions "Usman", "Kahuta", "Rawalpindi", "portfolio", or other project-related/personal terms → Use RAG only.
   - If the query is about cricket, players, teams, or general knowledge → Use WebScraping only.
   - If the query is a math problem (numbers with +, -, x, *, /) → Use Calculator only.
   - Use exactly one tool per query. Never call multiple tools for the same query.

2. Execution
   - Call the chosen tool exactly once per query. Do NOT retry the same tool.
   - Stop immediately after receiving the first valid tool result.
   - For Calculator:
       • If it produces a numeric result, return it directly.
       • Do not call Calculator again with the numeric result.
   - For RAG:
       • Do not generate multiple RAG calls with different variations (e.g., family, location, portfolio).
       • Only return the first RAG result as the final answer.
   - For WebScraping:
       • Only return factual cricket/sports info.
       • Do not call Calculator or RAG after WebScraping.

3. Output
   - Respond in 1–3 concise sentences.
   - Never output raw tool calls, JSON, or placeholders like <function=...>.
   - Only output the clean final answer.
   - If the tool cannot find an answer, respond with: "I don't know".
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", System_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

memory.chat_memory.add_message(SystemMessage(content=System_prompt))

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# === LLM ===
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "my-agent")

# === Agent ===
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=3,
    early_stopping_method="force",
)

# === Chat Loop ===
def chat_loop():
    print("Start chatting with the AI! Type 'exit' to end.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        memory.chat_memory.add_message(HumanMessage(content=user_input))

        try:
            response = agent_executor.invoke({"input": user_input})

            # Prefer the clean output
            output = response.get("output", "")

            # If output is missing, fall back to intermediate steps (tool return values)
            if not output and "intermediate_steps" in response:
                steps = response["intermediate_steps"]
                if steps and len(steps) > 0:
                    tool_result = steps[-1][1]  # Get the last tool's return value
                    output = str(tool_result)

            # Cleanup whitespace
            output = re.sub(r"\s+", " ", output).strip()

            memory.chat_memory.add_message(AIMessage(content=output))
            print("AI:", output)

        except Exception as e:
            print("AI: Error:", str(e))



if __name__ == "__main__":
    chat_loop()
