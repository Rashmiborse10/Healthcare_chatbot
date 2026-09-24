"""
app/llm/healthcare_assistant.py

The conversational orchestrator — now implemented as a LangGraph
`StateGraph` instead of a linear Python function chain.

Why a graph instead of a straight-line chain?
- Every guardrail is its own node with an explicit conditional edge, so the
  control flow (what runs, what short-circuits, what feeds what) is
  declared once as graph topology instead of buried in nested if/else.
- Nodes are independently testable: you can invoke any single node function
  with a partial state dict without running the rest of the pipeline.
- It's trivial to extend later (e.g. add a re-retrieval loop, a
  human-in-the-loop review node, or LangGraph's checkpointer for durable
  multi-turn memory) without restructuring the whole flow.

Graph topology:

    START
      │
      ▼
  emergency_check ──(blocked)──────────────────────────────┐
      │ continue                                            │
      ▼                                                      │
  injection_check ──(blocked)───────────────────────────────┤
      │ continue                                             │
      ▼                                                       │
  medical_check ──(blocked)──────────────────────────────────┤
      │ continue                                              │
      ▼                                                        │
  topic_check ──(blocked)─────────────────────────────────────┤
      │ continue                                               │
      ▼                                                         │
   retrieve ──(no relevant context)────────────────────────────┤
      │ continue                                                │
      ▼                                                         │
   generate                                                     │
      │                                                         │
      ▼                                                         │
  post_filter                                                   │
      │                                                         │
      ▼                                                         ▼
                            END
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from app.core.constants import NO_ANSWER_FOUND_MSG
from app.core.prompts import SYSTEM_PROMPT, RAG_ANSWER_TEMPLATE
from app.rag.rag_pipeline import RAGPipeline
from app.rag.retriever import RetrievedChunk
from app.rag.citation_generator import Citation
from app.llm.llm_factory import get_llm
from app.guardrails.prompt_injection import check_prompt_injection
from app.guardrails.emergency_detector import check_emergency
from app.guardrails.medical_guardrails import check_medical_guardrails
from app.guardrails.topic_validator import check_topic
from app.guardrails.response_filter import filter_response, ensure_sources_appended

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class AssistantResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    blocked_reason: Optional[str] = None  # "emergency" | "medical" | "injection" | "out_of_scope" | None
    grounded: bool = True  # False when we fell back / were blocked / were filtered


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AssistantState(TypedDict, total=False):
    """
    The shared state object every node reads from and writes into.
    `total=False` because most keys are only populated once their node runs.
    """
    question: str
    history: List[ChatTurn]

    # populated by guardrail nodes
    blocked_reason: Optional[str]
    blocked_message: Optional[str]
    # set by emergency_check when the message mentions an emergency-adjacent
    # topic informationally (e.g. "what are the symptoms of a heart
    # attack?") rather than reporting a live emergency — appended to the
    # final answer instead of blocking it.
    safety_note: Optional[str]

    # populated by `retrieve`
    context: str
    citations: List[Citation]
    chunks: List[RetrievedChunk]

    # populated by `generate` / `post_filter`
    raw_answer: str
    final_answer: str
    grounded: bool


def _format_history(history: List[ChatTurn], max_turns: int = 6) -> str:
    if not history:
        return "(no previous conversation)"
    recent = history[-max_turns:]
    return "\n".join(f"{turn.role.capitalize()}: {turn.content}" for turn in recent)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def node_emergency_check(state: AssistantState) -> AssistantState:
    result = check_emergency(state["question"])
    if result.is_emergency:
        logger.warning("[graph] emergency_check blocked the request.")
        return {"blocked_reason": "emergency", "blocked_message": result.message}
    if result.is_informational and result.safety_note:
        # Not blocked — proceed to RAG as normal, just carry the reminder
        # forward so post_filter can attach it to the final answer.
        return {"safety_note": result.safety_note}
    return {}


def node_injection_check(state: AssistantState) -> AssistantState:
    result = check_prompt_injection(state["question"])
    if result.is_injection:
        logger.warning("[graph] injection_check blocked the request.")
        return {"blocked_reason": "injection", "blocked_message": result.message}
    return {}


def node_medical_check(state: AssistantState) -> AssistantState:
    result = check_medical_guardrails(state["question"])
    if result.is_blocked:
        logger.warning("[graph] medical_check blocked the request.")
        return {"blocked_reason": "medical", "blocked_message": result.message}
    return {}


def node_topic_check(state: AssistantState) -> AssistantState:
    result = check_topic(state["question"])
    if not result.in_scope:
        logger.warning("[graph] topic_check blocked the request.")
        return {"blocked_reason": "out_of_scope", "blocked_message": result.message}
    return {}


def _make_retrieve_node(rag: RAGPipeline):
    def node_retrieve(state: AssistantState) -> AssistantState:
        context, citations, chunks = rag.retrieve(state["question"])
        if not context:
            logger.info("[graph] retrieve found no grounded context.")
            return {
                "blocked_reason": "no_context",
                "blocked_message": NO_ANSWER_FOUND_MSG,
                "citations": [],
                "chunks": [],
            }
        return {"context": context, "citations": citations, "chunks": chunks}
    return node_retrieve


def node_generate(state: AssistantState) -> AssistantState:
    prompt = RAG_ANSWER_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        chat_history=_format_history(state.get("history", [])),
        context=state["context"],
        question=state["question"],
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_answer = response.content or ""
    except Exception:
        logger.exception("[graph] generate node: LLM call failed.")
        return {
            "raw_answer": "",
            "blocked_reason": "llm_error",
            "blocked_message": "Something went wrong while generating a response. Please try again shortly.",
        }
    return {"raw_answer": raw_answer}


def node_post_filter(state: AssistantState) -> AssistantState:
    filtered = filter_response(state["raw_answer"])
    final_answer = ensure_sources_appended(filtered.text, state.get("citations", []))

    # Only attach the "seek help now if this is live" reminder to a genuine,
    # unmodified answer — not to a response that already got swapped out by
    # the safety filter.
    safety_note = state.get("safety_note")
    if safety_note and not filtered.was_modified:
        final_answer = f"{final_answer}{safety_note}"

    return {
        "final_answer": final_answer,
        "grounded": not filtered.was_modified,
        "blocked_reason": "medical" if filtered.was_modified else None,
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def _route_after_guardrail(state: AssistantState) -> Literal["blocked", "continue"]:
    return "blocked" if state.get("blocked_reason") else "continue"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(rag_pipeline: Optional[RAGPipeline] = None):
    """Construct and compile the LangGraph StateGraph for the assistant."""
    rag = rag_pipeline or RAGPipeline()
    graph = StateGraph(AssistantState)

    graph.add_node("emergency_check", node_emergency_check)
    graph.add_node("injection_check", node_injection_check)
    graph.add_node("medical_check", node_medical_check)
    graph.add_node("topic_check", node_topic_check)
    graph.add_node("retrieve", _make_retrieve_node(rag))
    graph.add_node("generate", node_generate)
    graph.add_node("post_filter", node_post_filter)

    graph.add_edge(START, "emergency_check")

    # Every guardrail node either short-circuits straight to END (blocked)
    # or falls through to the next stage.
    graph.add_conditional_edges(
        "emergency_check", _route_after_guardrail, {"blocked": END, "continue": "injection_check"}
    )
    graph.add_conditional_edges(
        "injection_check", _route_after_guardrail, {"blocked": END, "continue": "medical_check"}
    )
    graph.add_conditional_edges(
        "medical_check", _route_after_guardrail, {"blocked": END, "continue": "topic_check"}
    )
    graph.add_conditional_edges(
        "topic_check", _route_after_guardrail, {"blocked": END, "continue": "retrieve"}
    )
    graph.add_conditional_edges(
        "retrieve", _route_after_guardrail, {"blocked": END, "continue": "generate"}
    )
    graph.add_conditional_edges(
        "generate", _route_after_guardrail, {"blocked": END, "continue": "post_filter"}
    )
    graph.add_edge("post_filter", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public wrapper — same call signature as the pre-LangGraph version, so
# Person 2's FastAPI route doesn't need to change at all.
# ---------------------------------------------------------------------------
class HealthcareAssistant:
    """
    Thin, stateless wrapper around the compiled LangGraph app. Conversation
    history is still passed in by the caller (persisted in SQLite by Person
    2's service layer) rather than held here.
    """

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag = rag_pipeline or RAGPipeline()
        self._app = build_graph(self.rag)

    def answer(self, question: str, history: Optional[List[ChatTurn]] = None) -> AssistantResponse:
        initial_state: AssistantState = {
            "question": question,
            "history": history or [],
            "blocked_reason": None,
        }

        final_state: AssistantState = self._app.invoke(initial_state)

        if final_state.get("blocked_reason"):
            reason = final_state["blocked_reason"]
            # "no_context" / "llm_error" are internal-only reasons; the public
            # AssistantResponse.blocked_reason keeps the same 4 public values
            # as before, defaulting anything else to None with grounded=False.
            public_reason = reason if reason in {"emergency", "injection", "medical", "out_of_scope"} else None
            return AssistantResponse(
                answer=final_state.get("blocked_message", NO_ANSWER_FOUND_MSG),
                citations=final_state.get("citations", []) or [],
                blocked_reason=public_reason,
                grounded=False,
            )

        return AssistantResponse(
            answer=final_state.get("final_answer", NO_ANSWER_FOUND_MSG),
            citations=final_state.get("citations", []) or [],
            blocked_reason=None,
            grounded=final_state.get("grounded", True),
        )