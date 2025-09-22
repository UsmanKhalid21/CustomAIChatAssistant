import os

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain.schema import Document

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# for rag
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()



def load_pdf_file(data):
    loader = DirectoryLoader(
        data, 
        # multiple PDF files
        glob="*.pdf",
        loader_cls=PyPDFLoader
        )
    documents = loader.load()
    return documents

extracted_data = load_pdf_file("MyNotes")

# print(extracted_data)
# print(len(extracted_data))


def filter_docs(docs: List[Document]) -> List[Document]:

    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

minimal_docs = filter_docs(extracted_data)

# print(minimal_docs)

# Split the documents 

def text_Split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    chuck_texts = text_splitter.split_documents(minimal_docs)
    return chuck_texts

chuck_texts = text_Split(minimal_docs)



def do_embedding():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return embeddings

embedding = do_embedding()



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")



os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_ENV"] = PINECONE_ENV


pc = Pinecone(
    api_key=PINECONE_API_KEY,
)

index_name = "self-notes"


# create index if not already exists
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,    # must be lowercase + alphanumeric + dash
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )

# connect to index
index = pc.Index(index_name)

# insert docs
db = PineconeVectorStore.from_documents(
    documents=chuck_texts,
    embedding=embedding,
    index_name=index_name
)


# now retrieve docs
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

retrieved_docs = retriever.invoke("Who is usman khalid?")

print(retrieved_docs)

