# Mouham'IA محامي

**Your AI Legal Assistant** - A Retrieval-Augmented Generation system for legal documents with exact quote citations.

> *Mouham'IA* = محامي (Mouhami, "lawyer" in Arabic) + IA (AI)

## Features

- **Two document categories**: Law codes (shared) and user contracts (private)
- **Exact quote citations**: Returns verbatim quotes from source documents
- **Smart query routing**: Automatically detects whether to search laws, contracts, or both
- **Cloud-native**: Uses Pinecone for vector storage, OpenAI for embeddings and LLM

## Setup

### 1. Install dependencies

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install the package
uv pip install -e .
```

### 2. Configure API keys

Create a `.env` file in the project root:

```bash
# OpenAI API Key - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# Pinecone API Key - Get from https://app.pinecone.io
PINECONE_API_KEY=your-pinecone-api-key-here

# Pinecone Index Name (optional, defaults to "legal-rag")
PINECONE_INDEX_NAME=legal-rag
```

### 3. Initialize Pinecone index

```bash
legal-rag init
```

## Usage

### Ingest Documents

```bash
# Ingest law codes
legal-rag ingest-laws ./data/law_codes/

# Ingest a single law code PDF
legal-rag ingest-laws ./data/law_codes/civil_code.pdf

# Ingest user contracts
legal-rag ingest-contracts ./data/contracts/

# Ingest a single contract
legal-rag ingest-contracts ./data/contracts/my_contract.pdf
```

### Ask Questions

```bash
# Auto-detect which namespace to search
legal-rag ask "What are the termination clauses in my contract?"

# Force search in specific namespace
legal-rag ask "What does Article 1134 say?" --source law
legal-rag ask "What is my notice period?" --source contracts

# Search both namespaces
legal-rag ask "Does my contract comply with civil code requirements?" --source both
```

### Interactive Chat

```bash
legal-rag chat
```

### Check Index Status

```bash
legal-rag stats
```

## Response Format

The system returns answers with exact verbatim quotes:

```
ANSWER:
Your consulting contract allows for termination with 30 days written 
notice [1]. However, if termination is due to breach, the Civil Code 
requires you to provide formal notice first [2].

SOURCES:

[1] consulting_agreement_2024.pdf | Page 8, Section 12.1
    > "Either party may terminate this Agreement by providing thirty 
    > (30) calendar days' prior written notice to the other party."

[2] Code Civil | Article 1231
    > "Le débiteur est condamné, s'il y a lieu, au paiement de 
    > dommages et intérêts..."
```

## Project Structure

```
legal-rag/
├── src/legal_rag/
│   ├── cli.py              # Typer CLI commands
│   ├── config.py           # Settings via pydantic-settings
│   ├── embeddings.py       # OpenAI embeddings
│   ├── pinecone_store.py   # Vector database operations
│   ├── retriever.py        # Query routing and retrieval
│   ├── rag.py              # RAG orchestration
│   └── ingest/
│       ├── pdf_extractor.py  # PDF text extraction
│       └── chunker.py        # Smart text chunking
├── data/
│   ├── contracts/          # Place user contracts here
│   └── law_codes/          # Place law codes here
└── .env                    # API keys (not in git)
```

## Multi-User Upgrade Path

Currently designed for single-user. To upgrade to multi-user:

1. Change `user_contracts` namespace to `user_{user_id}`
2. Add authentication (Clerk, Auth0)
3. Pass user ID from JWT to all contract operations
4. Law codes namespace remains shared

