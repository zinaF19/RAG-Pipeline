# Retrieval Augmented Generation (RAG)

A hands-on workshop on Retrieval-Augmented Generation (RAG): building systems that overcome Large Language Model limitations by retrieving relevant information from your own documents and combining it with the user's query. You start from the core concepts, build a complete RAG pipeline over PDF research papers from scratch, then rebuild it with LangChain and Qdrant local mode. Both versions use local HuggingFace embeddings and the same configurable chat-model adapter, with Groq as the default. The first pipeline keeps chunking, retrieval and prompt construction manual; only provider transport is shared.

## Learning Objectives

By the end of this repository, you should be able to:

- Explain why RAG is needed compared to fine-tuning, and how it reduces hallucinations with transparent, source-grounded answers.
- Build a RAG pipeline from scratch: load and chunk PDFs, embed chunks, and retrieve them by cosine similarity.
- Store embeddings in a Qdrant vector database and run semantic similarity search.
- Connect a retriever to a Groq-hosted gpt-oss model to generate grounded answers.
- Build an interactive Q&A system over your own PDF documents.
- Convert and clean source documents with MarkItDown before indexing them.
- Recognise what the LangChain framework automates compared to a from-scratch build.

## Learning Path

Start with the concepts, build the pipeline from scratch, practise it on real documents, then see the same pipeline with LangChain, and finish with the real-world applications:

| File / Folder | Description |
|---|---|
| [**1 - RAG Concepts**](1_rag_concepts.md) | Foundational reading: what RAG is, how it compares to fine-tuning, and its key components. |
| [**2 - Build a RAG Pipeline from Scratch**](2_rag_from_scratch.ipynb) | Build retrieval and prompt construction by hand with no retrieval framework, so every stage is visible. |
| [**3 - From-Scratch RAG Exercise**](3_rag_from_scratch_exercise.ipynb) | Convert and clean a document with MarkItDown, then measure how cleaning changes retrieval, reusing the lesson's pipeline. |
| [**4 - RAG Pipeline with gpt-oss (LangChain)**](4_rag-pipeline-gpt-oss.ipynb) | The same pipeline built with LangChain, Qdrant, and a Groq-hosted gpt-oss model. |
| [**5 - Extend and Evaluate Exercise**](5_rag_exercise_notebook.ipynb) | Extend the LangChain pipeline with source citations, tune retrieval, then evaluate and compare configurations with an LLM judge. |
| [**6 - Real-World Applications**](6_rag_real_world_applications.md) | How RAG is used across industries (journalism, legal, healthcare, finance, and more). |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Documents**](documents/) | Sample PDFs used in the notebooks (an AI research paper and pharmaceutical documentation). |
| [**Assets**](assets/) | Workflow diagrams used in the notebooks. |
| [**Solutions**](solutions/) | Reference solutions. |
| [**.env.example**](.env.example) | Template for the environment variables you need to provide. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

Qdrant runs locally in process; no separate server or Docker container is needed.
Rebuild the indexes with the updated notebooks; old Chroma indexes are not converted.

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 or newer
and Python 3.13.17 or a newer 3.13 patch release, as defined in
[`.python-version`](.python-version).

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd ds-rag-pipeline`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in (`.venv/`).

```bash
cd <repo-name>
uv sync
```

> [!NOTE]
> The notebooks use a local HuggingFace embedding model (`all-mpnet-base-v2`). It downloads automatically on first run (around 400 MB), so the first notebook run takes a little longer. No HuggingFace account or token is needed.

---

### 5. Set Up Environment Variables

Copy the example, select `LLM_PROVIDER`, and fill in only the matching API key:

```bash
cp .env.example .env
```

- **Groq (primary):** Set `GROQ_API_KEY` from the [Groq Console](https://console.groq.com/keys); the default is `GROQ_MODEL=openai/gpt-oss-20b`.
- **Mistral (alternative):** Set `MISTRAL_API_KEY` from [Mistral AI](https://console.mistral.ai/api-keys); the default is `MISTRAL_MODEL=ministral-14b-latest`.
- **OpenRouter (optional):** Set `OPENROUTER_API_KEY`. The default `google/gemma-4-26b-a4b-it:free` has `openrouter/free` as a server-side backup, configured by `OPENROUTER_FALLBACK_MODEL`. Free accounts currently share [50 requests/day across free models](https://openrouter.ai/pricing); changing models does not reset this quota.
- **Google AI Studio (optional):** Set `GOOGLE_API_KEY` and `GOOGLE_MODEL=gemma-4-26b-a4b-it` for direct [Gemma access](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), independent of OpenRouter. Free [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) vary by project and model; use only non-sensitive exercise data on the free tier.

`llm_config.py` uses `init_chat_model` to select the native adapter. Prompts and application logic stay the same; `*_MODEL` overrides the selected model. Calls have a 30-second timeout, no automatic retries, and a bounded output budget. A different model must support the tools or structured outputs the exercise needs. Restart the kernel after editing `.env`.

> [!CAUTION]
> Your `.env` file holds a secret and must never be committed. Only `.env.example`, with placeholder values, belongs in the repository.

---

### 6. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**LangChain Documentation**](https://docs.langchain.com/oss/python/langchain/overview): The official LangChain guide for building LLM applications.
- [**Qdrant**](https://qdrant.tech/documentation/): Open-source vector database for storing embeddings and running similarity search.
- [**Groq Console**](https://console.groq.com/playground): Where you create your API key and test models in the playground.
- [**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**](https://arxiv.org/abs/2005.11401): The original RAG paper (Lewis et al., 2020).
- [**Retrieval-Augmented Generation for Large Language Models: A Survey**](https://arxiv.org/abs/2312.10997): A broad survey of RAG techniques (Gao et al., 2023).
