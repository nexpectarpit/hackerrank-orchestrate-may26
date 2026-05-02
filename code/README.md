# HackerRank Orchestrate AI Support Agent

This directory contains the implementation of the AI support agent for the HackerRank Orchestrate Hackathon. 
The agent provides automated, intelligent triage for support tickets across HackerRank, Claude, and Visa.

## Prerequisites

- Python 3.9+
- A `.env` file with your `GROQ_API_KEY` (see `.env.example`).

## Quick Start

### 1. Install Dependencies
Set up your virtual environment and install the required packages:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and add your Groq API key:
```bash
cp .env.example .env
# Edit .env and add GROQ_API_KEY=gsk_yourkeyhere
```

### 3. Run the Agent
Execute the main script to process the tickets located in `../support_tickets/support_tickets.csv`:
```bash
python main.py
```

## Outputs
- **`../support_tickets/output.csv`**: The fully processed support tickets, formatted strictly to the problem statement schema.
- **`execution_log.txt`**: A clean, structural log detailing the agent's step-by-step reasoning and decisions for each ticket.

## Architecture Highlights
- **Strict Routing Enforcement**: Ensures 100% accuracy in company assignment when explicitly provided.
- **Sensitivity Safety Switch**: High-risk issues (fraud, billing, account access) are intercepted and forcefully escalated to prevent hallucinations.
- **Modular Pipeline**: Code is structured into `safety`, `router`, `retriever`, `generator`, and `critic` modules for clear separation of concerns.
- **FAISS Semantic Retrieval**: Documents are ingested locally and semantically searched using `all-MiniLM-L6-v2` embeddings.
