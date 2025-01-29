# project_documind
# DocuMind AI 🧠

**Intelligent Document Analysis System with Domain-Specific AI**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)

## Abstract 📝
DocuMind AI is an advanced document analysis tool combining OCR, NLP, and domain-specific AI to process medical, legal, and general documents. Leveraging GPT-4 and LangChain, it provides:

- Hybrid text extraction (native PDF + OCR fallback)
- Domain-optimized analysis (medical/legal/general)
- Source-aware citations with page references
- Secure local processing pipeline
- Multi-document cross-referencing

## Features ✨
- ⚕️ Medical diagnosis extraction
- ⚖️ Legal clause analysis
- 🔍 Scanned document support
- 📑 Multi-PDF batch processing
- 🔒 Confidentiality-preserving architecture

## Quick Start 🚀

```bash
# Clone and setup
git clone https://github.com/yourusername/documind-ai.git
cd documind-ai

# Install system dependencies
# For Mac:
brew install poppler tesseract && export TESSERACT_PATH="/opt/homebrew/bin/tesseract"

# For Linux:
sudo apt-get update && sudo apt-get install -y poppler-utils tesseract-ocr libtesseract-dev

# Python setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
...............................................................................

# Run (after adding OpenAI API key to .env)
echo "OPENAI_API_KEY=your_key" > .env && streamlit run app.py

brew install poppler tesseract && export TESSERACT_PATH="/opt/homebrew/bin/tesseract"; python -m venv venv && source venv/bin/activate && pip install -r requirements.txt; echo "OPENAI_API_KEY=your_key" > .env && streamlit run app.py


