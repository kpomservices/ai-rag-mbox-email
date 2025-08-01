import os
import warnings

# Option 1: Enable LangSmith monitoring (replace with your actual key)
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-document-qa"

# Option 2: Disable LangSmith (uncomment these lines instead)
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Prepare the document
arxiv_contents = "Your raw text or email content goes here"
doc = Document(page_content=arxiv_contents, metadata={"source": "local"})

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents([doc])

# 3. Create vector store
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=GPT4AllEmbeddings()
)

# 4. Load local LLM
llm = GPT4All(
    model="models/q4_0-orca-mini-3b.gguf",
    max_tokens=2048,
)

# 5. Set up retriever
retriever = vector_store.as_retriever()

# 6. Load prompt from LangChain Hub
try:
    rag_prompt = hub.pull("rlm/rag-prompt")
    print("Successfully loaded prompt from LangChain Hub")
except Exception as e:
    print(f"Hub access failed: {e}")
    print("Using fallback prompt...")
    from langchain_core.prompts import ChatPromptTemplate
    rag_prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context below:\n\n"
        "Context: {context}\n\n" 
        "Question: {question}\n\n"
        "Answer:"
    )

# 7. Define a simple formatting function
def format_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 8. Create the full RAG chain
qa_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 9. Run a test query
print("Running query...")
response = qa_chain.invoke("What does the document say about awareness?")
print("Response:", response)

# If using LangSmith, you can view traces at: https://smith.langchain.com/