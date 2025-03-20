import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
import tempfile
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set page configuration
st.set_page_config(
    page_title="Turbine Maintenance Assistant",
    page_icon="🔧",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e9ecef;
    }
    .chat-message.bot {
        background-color: #d2e3fc;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'generated' not in st.session_state:
    st.session_state.generated = []
if 'past' not in st.session_state:
    st.session_state.past = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

# Display header and description
st.title("🔧 Turbine Maintenance Assistant")
st.markdown("""
This assistant specializes in providing information about maintenance and troubleshooting for wind and gas turbines.
Upload your technical documents, manuals, or guides to customize the knowledge base.
""")


# Function to load and process documents
def process_documents(uploaded_files):
    with st.spinner("Processing documents..."):
        # Create a temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()

        # Save uploaded files to temporary directory
        file_paths = []
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_file_path)

        # Load documents
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        return chunks


# Function to create vector store
def create_vectorstore(chunks):
    with st.spinner("Creating knowledge base..."):
        api_key = "AIzaSyDQInx8mp8C5RHxpQKYl1ujorQZ1w69OtA"
        genai.configure(api_key=api_key)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore


# Function to create conversation chain
def create_conversation_chain(vectorstore):
    api_key = st.session_state.api_key
    genai.configure(api_key=api_key)

    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        top_p=0.85,
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create prompt template
    template = """You are an expert technical assistant specializing in wind and gas turbines.
    You provide factual, reliable information about turbine maintenance, troubleshooting, and operations.

    Use the following context to answer the user's question. If you don't know the answer, don't make something up - just say you don't have enough information.

    When discussing technical procedures, be clear and precise.
    For troubleshooting, list steps in a methodical order.
    If a question requires safety considerations, always emphasize safety protocols first.

    Context: {context}

    Chat History: {chat_history}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    return conversation_chain


# API key handling
with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input("Enter your Google AI API key:", type="password")
    if api_key:
        st.session_state.api_key = api_key
        genai.configure(api_key=api_key)

    # Document uploading
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files about turbine maintenance",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )

    if uploaded_files and api_key and st.button("Process Documents"):
        chunks = process_documents(uploaded_files)
        st.session_state.vectorstore = create_vectorstore(chunks)
        st.session_state.conversation_chain = create_conversation_chain(st.session_state.vectorstore)
        st.success(f"Processed {len(uploaded_files)} documents. Knowledge base created!")

    # Sample questions
    st.subheader("Sample Questions")
    sample_questions = [
        "What are common causes of vibration in gas turbines?",
        "How do I troubleshoot a pitch system failure in a wind turbine?",
        "What's the maintenance schedule for a GE gas turbine?",
        "How do weather conditions affect wind turbine performance?"
    ]

    for question in sample_questions:
        if st.button(question):
            st.session_state.user_input = question


# Function to display chat message
def display_chat_message(message, is_user=False):
    if is_user:
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar">🔧</div>
            <div class="message">{message}</div>
        </div>
        """, unsafe_allow_html=True)


# Main chat interface
st.subheader("Chat with the Turbine Maintenance Assistant")

# Display chat messages from history
for i in range(len(st.session_state.past)):
    display_chat_message(st.session_state.past[i], is_user=True)
    display_chat_message(st.session_state.generated[i])

# User input
user_input = st.text_input("Ask a question about turbine maintenance:", key="user_input")

# Check if the user has uploaded documents and entered an API key
if not api_key:
    st.warning("Please enter your Google AI API key in the sidebar.")
elif st.session_state.vectorstore is None:
    st.warning("Please upload and process documents to create a knowledge base.")
elif user_input:
    if st.session_state.conversation_chain:
        with st.spinner("Thinking..."):
            # Get response from conversation chain
            response = st.session_state.conversation_chain.invoke({
                "question": user_input,
            })
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response["answer"])

            # Clear the input box
            st.session_state.user_input = ""

            # Rerun to update the chat display
            st.experimental_rerun()

# Instructions for deployment
st.sidebar.subheader("Deployment Instructions")
st.sidebar.markdown("""
1. Save this code as `app.py`
2. Create a `requirements.txt` file with needed dependencies
3. Create a GitHub repository with both files
4. Deploy on Streamlit.io by connecting to your repository
""")

# Requirements for requirements.txt
st.sidebar.code("""
streamlit
langchain
langchain-community
langchain-google-genai
google-generativeai
faiss-cpu
PyPDF2
tiktoken
""", language="text")