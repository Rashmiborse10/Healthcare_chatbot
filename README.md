# Person 1 — RAG & AI Engineer Module

This covers `app/rag/`, `app/llm/`, `app/guardrails/`, and `app/core/` from the
Healthcare Knowledge Assistant project.

## Folder structure

```
app/
├── core/
│   ├── constants.py          # paths, chunk size, top_k, guardrail messages, keyword lists
│   └── prompts.py            # system prompt + RAG prompt templates
│
├── rag/
│   ├── data_loader.py        # Phase 1: load PDF/DOCX/TXT, attach metadata
│   ├── text_splitter.py      # Phase 2: chunk documents (RecursiveCharacterTextSplitter)
│   ├── embeddings.py         # Phase 2: all-MiniLM-L6-v2 embedding singleton
│   ├── vector_store.py       # Phase 2: build/load/persist FAISS index
│   ├── retriever.py          # Phase 3: top-k similarity search + score threshold
│   ├── citation_generator.py # Phase 5: RetrievedChunk -> structured Citation objects
│   └── rag_pipeline.py       # orchestrates the above; ingest_knowledge_base() + RAGPipeline
│
├── llm/
│   ├── llm_factory.py            # Phase 4: chat model singleton, served via OpenRouter
│   └── healthcare_assistant.py   # LangGraph StateGraph: guardrails -> retrieval -> LLM -> guardrails
│
└── guardrails/               # Phase 6
    ├── emergency_detector.py   # chest pain / stroke / suicidal ideation -> safety message
    ├── prompt_injection.py     # "ignore instructions", "reveal system prompt", etc.
    ├── medical_guardrails.py   # blocks diagnosis / dosage / prescription requests
    ├── topic_validator.py      # out-of-scope detection (keyword heuristic + LLM fallback)
    └── response_filter.py      # post-generation safety net + guarantees "Sources:" line
```

## Data flow — LangGraph `StateGraph`

`healthcare_assistant.py` no longer implements the pipeline as a straight-line
Python function. It builds a **LangGraph `StateGraph`** where every stage is a
node and every guardrail is a conditional edge that can short-circuit
straight to `END`:

```
START
  │
  ▼
emergency_check ──(blocked)───────────────────────────────┐
  │ continue                                                │
  ▼                                                          │
injection_check ──(blocked)────────────────────────────────┤
  │ continue                                                 │
  ▼                                                           │
medical_check ──(blocked)───────────────────────────────────┤
  │ continue                                                  │
  ▼                                                            │
topic_check ──(blocked)─────────────────────────────────────  ┤
  │ continue                                                   │
  ▼                                                             │
retrieve ──(no relevant context found)───────────────────────  ┤
  │ continue                                                    │
  ▼                                                              │
generate  (Gemini call, RAG_ANSWER_TEMPLATE)                     │
  │                                                              │
  ▼                                                              │
post_filter (safety-net regex scan + "Sources:" line)            │
  │                                                              │
  ▼                                                              ▼
                          END
```

**State object** (`AssistantState`, a `TypedDict`): `question`, `history`,
`blocked_reason`, `blocked_message`, `context`, `citations`, `chunks`,
`raw_answer`, `final_answer`, `grounded`. Each node reads what it needs from
state and returns only the keys it updates — LangGraph merges partial
updates into the running state automatically.

**Why this is better than a flat chain:**
- Each guardrail is an independently testable node function
  (`node_emergency_check`, `node_injection_check`, ...) you can unit-test by
  calling it directly with a partial state dict.
- The control flow is declared once as graph topology
  (`add_conditional_edges`) rather than nested `if/else` — easy to see at a
  glance which stage can short-circuit and where it goes.
- Easy to extend later: add a re-retrieval / query-rewrite loop, a
  human-in-the-loop review node, or swap in LangGraph's `MemorySaver`
  checkpointer for durable multi-turn state without restructuring anything.
- `build_graph()` is compiled once per `HealthcareAssistant` instance
  (`self._app = build_graph(...)`), so repeated `.answer()` calls just
  invoke the already-compiled graph.

## How Person 2 (Backend/API) should integrate this

```python
from app.llm.healthcare_assistant import HealthcareAssistant, ChatTurn

assistant = HealthcareAssistant()

history = [ChatTurn(role="user", content="What is metabolic syndrome?"),
           ChatTurn(role="assistant", content="...")]

result = assistant.answer("What blood pressure counts as high risk?", history=history)

result.answer            # str, ready to show the user (includes "Sources:" line)
result.citations         # List[Citation] -> store in the `citations` table
result.blocked_reason    # None | "emergency" | "injection" | "medical" | "out_of_scope"
result.grounded          # False if the answer wasn't grounded in the KB
```

## Running ingestion

1. Drop the 4 knowledge-base files into `data/` (PDF, DOCX, or TXT).
2. `pip install -r requirements-rag.txt`
3. Copy `.env.rag.example` -> `.env` and set `OPENROUTER_API_KEY`.
4. `python -m scripts.ingest` (add `--reset` to rebuild from scratch).
5. The FAISS index is persisted under `vector_db/`.

## Notes on the guardrail design

- All pre-generation guardrails (`emergency_detector`, `prompt_injection`,
  `medical_guardrails`, `topic_validator`) run **before** any retrieval or
  LLM call, so a blocked message never even reaches the knowledge base or
  Gemini — this saves cost/latency and guarantees the refusal wording is
  exactly what was specified, not something the LLM paraphrased.
- `response_filter` is a **second, independent layer** that scans the LLM's
  actual output for diagnostic/prescriptive phrasing before it reaches the
  user — defense in depth, since no prompt-only approach is 100% reliable.
- The retriever's similarity threshold (`SIMILARITY_SCORE_THRESHOLD` in
  `core/constants.py`) is what prevents hallucination on genuinely
  out-of-knowledge-base questions: if nothing scores high enough, the
  pipeline returns `NO_ANSWER_FOUND_MSG` instead of asking Gemini to answer
  from context that isn't actually relevant.
