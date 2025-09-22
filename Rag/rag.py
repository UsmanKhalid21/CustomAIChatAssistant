import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize embeddings
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

# 2. Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENV"] = PINECONE_ENV

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "self-notes"

# 3. Connect retriever directly (no history-aware reformulation)
retriever = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
).as_retriever(
    search_kwargs={"k": 3},
    include_metadata=True
)

# 4. Setup LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# 5. System prompt for QA

qa_system_prompt = (
    "You are an AI assistant specialized in question-answering tasks. "
    "Use the provided context below to answer the user's question clearly and concisely.\n\n"
    "Guidelines:\n"
    "• Always use whatever relevant context is provided.\n"
    "• If only partial information is found, summarize what is available.\n"
    "• If absolutely no relevant information is found, respond with 'I don't know based on the provided context'.\n"
    "• Keep responses to a maximum of three sentences.\n"
    "• Focus on the user's intent, even if their question contains spelling errors or is incomplete.\n"
    "• Provide direct, relevant answers without unnecessary elaboration.\n\n"
    "{context}"
)


# 6. QA Prompt Template
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 7. Chain setup
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -------------------------------
# Functions
# -------------------------------
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=result["answer"]))


def rag_query(query: str, chat_history) -> str:
    try:
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        return result.get("answer", "No answer found in context.")
    except Exception as e:
        return f"Error in rag_query: {str(e)}"


# Main function to start the continual chat
if __name__ == "__main__":
    continual_chat()
