# Contribution Rules

This document defines the working rules for the FPT AI Chatbot project.
All members must follow these rules strictly.

---

## 1. Branch Strategy

We use a single GitHub repository with the following branches:

- `main`

  - Stable version only
  - Always runnable
  - No direct push allowed

- `ai`

  - RAG, LLM, embeddings, ML models
  - Data processing and evaluation

- `backend`

  - API, server logic, database, authentication

- `voice`

  - Speech-to-Text (STT)
  - Text-to-Speech (TTS)
  - Audio pipeline

- `frontend`
  - Web UI
  - User interaction

---

## 2. General Rules

- Each member works **only on their assigned branch**
- **Push at least once per day**
- Commit messages must be clear and meaningful
- Do NOT push directly to `main`
- No push = no work done

---

## 3. Daily Workflow

1. Pull the latest code from your branch:
   ```bash
   git pull origin <your-branch>
   ```
