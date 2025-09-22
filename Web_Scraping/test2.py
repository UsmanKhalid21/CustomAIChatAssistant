import os
from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
from pydantic import Field

from langchain_text_splitters import CharacterTextSplitter

# === Load API keys ===
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Scrape Website ===
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY, api_url="https://api.firecrawl.dev")

# Scrape full homepage
result_home = firecrawl.scrape(
    "https://www.icc-cricket.com/",
    formats=["markdown"],
    only_main_content=False,  # get all sections
    timeout=120000
)

# Scrape men’s batting rankings separately
result_men_batting = firecrawl.scrape(
    "https://www.icc-cricket.com/rankings/mens/player-rankings/test/batting",
    formats=["markdown"],
    only_main_content=False,
    timeout=12000
)

# Combine content
full_text = result_home.markdown + "\n\n" + result_men_batting.markdown
# === Chunking helper ===
# splitting into chunks

def chunk_text(text, chunk_size=500):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return text_splitter.split_text(text)


chunks = chunk_text(full_text)

# Convert chunks to Document objects
documents = [Document(page_content=chunk) for chunk in chunks]


# === Setup LLM (Groq) ===
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# === Create a simple static retriever ===

# === Create a simple static retriever ===
class StaticRetriever(BaseRetriever):
    """Retriever that just returns pre-split static documents."""

    docs: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs[:5]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.docs[:5]



# Export retriever for reuse
retriever = StaticRetriever(docs=documents)


# ---- Contextualize prompt ----
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# ---- QA prompt ----
qa_system_prompt = """
You are a question answering assistant.
Answer the user's question based on the following website text:

Website Content:
{context}

Answer clearly and concisely.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)




def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Pass raw text, not embeddings
        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result["answer"]))



def web_query(query: str, chat_history) -> str:
    try:
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        return result.get("answer", "No answer found in context.")
    except Exception as e:
        return f"Error in rag_query: {str(e)}"
    



# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()