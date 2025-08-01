import os
import warnings
from tqdm import tqdm
from bs4 import BeautifulSoup
from mailbox import mbox as MboxReader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangSmith configuration (optional)
os.environ["LANGSMITH_API_KEY"] = "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"  # Replace with your key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "arxiv-email-rag"

# Option to disable LangSmith (uncomment if you don't want monitoring)
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

def process_mbox_emails(path, emails_to_process=10):
    """
    Process emails from mbox file and extract text content
    """
    print(f"Processing emails from: {path}")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mbox file not found: {path}")
    
    try:
        mbox = MboxReader(path)
        current_mails = 0
        arxiv_contents = ""
        processed_emails = []
        
        print(f"Processing up to {emails_to_process} emails...")
        
        for idx, message in tqdm(enumerate(mbox), desc="Processing emails"):
            try:
                # Get email metadata
                subject = message.get('Subject', 'No Subject')
                sender = message.get('From', 'Unknown Sender')
                date = message.get('Date', 'Unknown Date')
                
                # Get payload
                payload = message.get_payload(decode=True)
                if payload:
                    current_mails += 1
                    
                    # Stop if we've processed enough emails
                    if current_mails > emails_to_process:
                        break
                    
                    # Parse HTML content
                    try:
                        # Handle encoding
                        if isinstance(payload, bytes):
                            payload_str = payload.decode('utf-8', errors='ignore')
                        else:
                            payload_str = str(payload)
                        
                        soup = BeautifulSoup(payload_str, 'html.parser')
                        body_text = soup.get_text().replace('"','').replace("\n", " ").replace("\t", " ").strip()
                        
                        # Clean up excessive whitespace
                        body_text = ' '.join(body_text.split())
                        
                        # Add email separator and metadata
                        email_content = f"\\n\\n--- Email {current_mails} ---\\n"
                        email_content += f"Subject: {subject}\\n"
                        email_content += f"From: {sender}\\n"
                        email_content += f"Content: {body_text}\\n"
                        
                        arxiv_contents += email_content
                        
                        # Store processed email info
                        processed_emails.append({
                            'index': current_mails,
                            'subject': subject,
                            'sender': sender,
                            'date': date,
                            'content_length': len(body_text)
                        })
                        
                        print(f"Processed email {current_mails}: {subject[:50]}...")
                        
                    except Exception as e:
                        print(f"Error processing email {current_mails} content: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing email {idx}: {e}")
                continue
        
        print(f"Successfully processed {current_mails} emails")
        print(f"Total content length: {len(arxiv_contents)} characters")
        
        return arxiv_contents, processed_emails
        
    except Exception as e:
        print(f"Error opening mbox file: {e}")
        raise

def create_rag_pipeline_from_emails(mbox_path, emails_to_process=10):
    """
    Create RAG pipeline from mbox emails
    """
    print("=== Email Processing Phase ===")
    
    # 1. Process emails from mbox file
    arxiv_contents, processed_emails = process_mbox_emails(mbox_path, emails_to_process)
    
    if len(arxiv_contents.strip()) < 100:
        raise ValueError("Insufficient email content extracted. Check your mbox file.")
    
    # Create document with metadata
    doc = Document(
        page_content=arxiv_contents, 
        metadata={
            "source": "arxiv_emails", 
            "email_count": len(processed_emails),
            "total_chars": len(arxiv_contents)
        }
    )
    
    print("\\n=== Document Splitting Phase ===")
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Larger chunks for email content
        chunk_overlap=100,  # More overlap to maintain context
        separators=["\\n\\n--- Email", "\\n\\n", "\\n", ". ", " "]
    )
    all_splits = text_splitter.split_documents([doc])
    
    print(f"Document split into {len(all_splits)} chunks")
    print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in all_splits) // len(all_splits)} characters")
    
    print("\\n=== Vector Store Creation Phase ===")
    
    # 3. Create vector store
    try:
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=GPT4AllEmbeddings(),
            persist_directory="./arxiv_chroma_db"
        )
        print("Vector store created successfully")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise
    
    print("\\n=== Model Loading Phase ===")
    
    # 4. Load local LLM
    model_path = "models/q4_0-orca-mini-3b.gguf"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        llm = GPT4All(
            model=model_path,
            max_tokens=2048,
            n_threads=4,
            device='cpu'
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # 5. Set up retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}  # Return top 5 relevant chunks
    )
    
    # 6. Load prompt
    try:
        rag_prompt = hub.pull("rlm/rag-prompt")
        print("RAG prompt loaded from hub")
    except Exception as e:
        print(f"Using fallback prompt: {e}")
        from langchain_core.prompts import ChatPromptTemplate
        rag_prompt = ChatPromptTemplate.from_template(
            "Based on the ArXiv email content below, answer the question accurately.\\n\\n"
            "Email Content: {context}\\n\\n"
            "Question: {question}\\n\\n"
            "Answer:"
        )
    
    # 7. Define formatting function
    def format_documents(docs):
        if not docs:
            return "No relevant email content found."
        return "\\n\\n".join([doc.page_content for doc in docs])
    
    # 8. Create the RAG chain
    qa_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever, processed_emails

def main():
    """
    Main function to run the email RAG pipeline
    """
    # Configuration
    mbox_path = "Mail/Category Promotions.mbox"
    emails_to_process = 10
    
    try:
        print("Starting ArXiv Email RAG Pipeline...")
        
        # Create pipeline
        qa_chain, retriever, processed_emails = create_rag_pipeline_from_emails(
            mbox_path, 
            emails_to_process
        )
        
        print("\\n=== Pipeline Ready! ===")
        print(f"Processed {len(processed_emails)} emails")
        
        # Display processed emails summary
        print("\\nProcessed Emails:")
        for email in processed_emails[:5]:  # Show first 5
            print(f"  {email['index']}: {email['subject'][:60]}...")
        
        print("\\n=== Testing Queries ===")
        
        # Test queries
        test_questions = [
            "What papers or topics are mentioned in these emails?",
            "What does the content say about machine learning?",
            "Are there any research papers about neural networks?",
            "What conferences or journals are mentioned?",
            "What are the main research areas covered?"
        ]
        
        for question in test_questions:
            print(f"\\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")
            
            # Test retrieval
            docs = retriever.invoke(question)
            print(f"Found {len(docs)} relevant email chunks")
            
            # Get answer
            try:
                response = qa_chain.invoke(question)
                print(f"Answer: {response}")
            except Exception as e:
                print(f"Error generating answer: {e}")
        
        return qa_chain, retriever, processed_emails
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return None, None, None

if __name__ == "__main__":
    qa_chain, retriever, processed_emails = main()