import streamlit as st

# Import the openai library
import openai # type: ignore

# Set page config FIRST
st.set_page_config(
    page_title="DocuMind AI", 
    page_icon="🧠", 
    layout="wide"
)

# Now other imports
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from pdf2image import convert_from_bytes # type: ignore
import pytesseract # type: ignore
import io
import base64
from streamlit_extras.stylable_container import stylable_container # type: ignore

# Configure M1 paths
POPPLER_PATH = "/opt/homebrew/bin"
TESSERACT_PATH = "/opt/homebrew/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Load environment variables
load_dotenv()

# Set the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #F5F7FB; }
    .stSelectbox, .stFileUploader { border-radius: 10px; padding: 15px; }
    .sidebar .sidebar-content { background-color: #2E4053; }
    h1 { color: #2E4053; border-bottom: 3px solid #48A9A6; }
    .stSpinner > div { border-color: #48A9A6 transparent transparent transparent; }
    .success-message { color: #48A9A6; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Domain configurations
DOMAIN_SETTINGS = {
    "medical": {
        "prompt": """You are a medical document analysis assistant. Follow these rules:
1. Base responses STRICTLY on the provided document content
2. Format diagnoses as bullet points with [Page X] references
3. Never refuse to answer if document contains relevant info
4. If unsure, say "Per document: [quote relevant text]"

Question: {question}
Document Content: {context}
Medical Analysis:""",
        "temperature": 0.2,
        "icon": "⚕️",
        "color": "#48A9A6"
    },
    "legal": {
        "prompt": """Analyze legal documents with precision:
1. Cite relevant clauses/sections with page numbers
2. Highlight potential liabilities
3. Identify key parties and obligations
4. Maintain strict confidentiality

Question: {question}
Document Content: {context}
Legal Analysis:""",
        "temperature": 0.3,
        "icon": "⚖️",
        "color": "#2E4053"
    },
    "general": {
        "prompt": """Comprehensive document analysis:
{context}
Question: {question}
Detailed Answer:""",
        "temperature": 0.4,
        "icon": "📄",
        "color": "#C44536"
    }
}

def extract_text_from_pdf(pdf_file):
    """Extract text with hybrid OCR/text extraction"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and len(page_text) > 100:
                text += page_text
            else:
                images = convert_from_bytes(
                    pdf_bytes,
                    first_page=page_number,
                    last_page=page_number,
                    poppler_path=POPPLER_PATH
                )
                if images:
                    page_text = pytesseract.image_to_string(images[0])
                    text += page_text
        return text
    except Exception as e:
        st.error(f"Text extraction failed: {str(e)}")
        return ""

def process_pdfs(pdf_files):
    """Process PDFs with enhanced metadata tracking"""
    all_chunks = []
    for pdf_file in pdf_files:
        with st.spinner(f"Processing {pdf_file.name}..."):
            text = extract_text_from_pdf(pdf_file)
            if not text:
                continue
                
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": pdf_file.name,
                    "chunk_id": f"{pdf_file.name}_chunk{i+1}"
                })
    
    if not all_chunks:
        st.error("No text extracted from any of the PDFs.")
        return None
    
    with st.spinner("Creating search index..."):
        embeddings = OpenAIEmbeddings()
        if len(all_chunks) == 0:
            st.error("No chunks were created from the extracted text.")
            return None
        
        return FAISS.from_texts(
            texts=[chunk["text"] for chunk in all_chunks],
            embedding=embeddings,
            metadatas=[{"source": chunk["source"], "chunk_id": chunk["chunk_id"]} 
                      for chunk in all_chunks]
        )

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    selected_domain = st.selectbox(
        "Select Analysis Domain",
        options=list(DOMAIN_SETTINGS.keys()),
        format_func=lambda x: f"{DOMAIN_SETTINGS[x]['icon']} {x.capitalize()}"
    )
    st.markdown("---")
    st.caption("ℹ️ Supported file types: PDF")
    st.caption("🔍 Requires OCR dependencies: `brew install poppler tesseract`")

# Main Content
st.title(f"{DOMAIN_SETTINGS[selected_domain]['icon']} DocuMind AI - {selected_domain.capitalize()} Analysis")
st.markdown("### Intelligent Document Processing & Analysis")

# File Upload
with st.container():
    pdf_files = st.file_uploader(
        "Upload Documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDFs for comprehensive analysis"
    )

# Document Processing
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

if pdf_files:
    with st.status(f"Processing {len(pdf_files)} documents...", expanded=True) as status:
        st.session_state.knowledge_base = process_pdfs(pdf_files)
        status.update(label="Processing Complete!", state="complete", expanded=False)

# Query Interface
if st.session_state.knowledge_base:
    with st.form("query_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            user_question = st.text_input(
                "Enter your question:",
                placeholder=f"Ask a {selected_domain} question about the documents...",
                label_visibility="collapsed"
            )
        with col2:
            submit_btn = st.form_submit_button("Analyze", type="primary")
        
        if submit_btn and user_question:
            with st.spinner("Analyzing documents..."):
                try:
                    docs = st.session_state.knowledge_base.similarity_search(user_question, k=5)
                    domain_config = DOMAIN_SETTINGS[selected_domain]
                    
                    qa_prompt = PromptTemplate(
                        template=domain_config["prompt"],
                        input_variables=["context", "question"]
                    )
                    
                    messages = [{"role": "system", "content": qa_prompt.format(context=' '.join(doc.page_content for doc in docs), question=user_question)}]
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=domain_config["temperature"],
                        max_tokens=500
                    )
                    
                    output_text = response.choices[0].message["content"]
                    
                    with stylable_container(
                        key="response_box",
                        css_styles=f"""
                            {{
                                border: 2px solid {domain_config['color']};
                                border-radius: 10px;
                                padding: 20px;
                                margin: 10px 0;
                            }}
                        """
                    ):
                        st.markdown(f"**Analysis Results** ({domain_config['icon']} {selected_domain.capitalize()})")
                        st.write(output_text)
                    
                    with st.expander("📚 Source References", expanded=True):
                        sources = {doc.metadata["source"] for doc in docs}
                        for source in sources:
                            st.markdown(f"`📄 {source}`")
                            
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("🧠 DocuMind AI v1.2 | Secure Document Analysis | Powered by LangChain & OpenAI")
