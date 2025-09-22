import os
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from firecrawl.types import Document

# === Load API keys ===
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Scrape Website ===
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY, api_url="https://api.firecrawl.dev")

class JsonSchema(BaseModel):
    company_mission: str
    supports_sso: bool
    is_open_source: bool
    is_in_yc: bool

result = firecrawl.scrape(
    'https://www.icc-cricket.com/',
    formats=[{
      "type": "json",
      "schema": JsonSchema
    }],
    only_main_content=False,
    timeout=120000
)

print(result)

page_text = result.markdown

# === Setup LLM (Groq) ===
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# === Ask a question directly ===
template = """
You are a question answering assistant. 
Answer the user's question based on the following website text:

Website Content:
{context}

Question:
{question}

Answer clearly and concisely.
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

question = "Who is the current captain of the Indian cricket team?"
final_prompt = prompt.format(context=page_text, question=question)

response = llm.invoke(final_prompt)

print("Q:", question)
print("A:", response.content)


