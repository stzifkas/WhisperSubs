"""LangGraph-based translation pipeline.

Graph topology:
  [START] → refine_node → (conditional) → translate_node → [END]
                                ↓ (no target language)
                              [END]
"""
import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from . import config
from .confidence import confidence_level
from .span_classifier import annotate_transcript, classify_spans

logger = logging.getLogger(__name__)


class SessionContext(TypedDict):
    summary: str
    glossary: dict[str, str]
    recent_pairs: list[tuple[str, str]]
    covered_up_to: int
    target_language: str
    _updating: bool


class TranslationState(TypedDict):
    raw_transcript: str
    target_language: str
    whisper_language: str
    session_context: SessionContext
    confidence: float | None           # geometric mean token probability; None = unknown
    logprobs: list[dict] | None        # raw token logprobs from Realtime API; None = unavailable
    refinement_low_threshold: float    # spans below this are always repaired
    refinement_high_risk_threshold: float  # high-risk spans below this are repaired
    correction_aggressiveness: str     # "low" | "medium" | "high"
    refined_source: str                # populated by refine_node
    translation: str                   # populated by translate_node



class RefinementOutput(BaseModel):
    source: str



_refine_llm = ChatOpenAI(
    model=config.TRANSLATION_MODEL,
    temperature=0.2,
    streaming=False,
    api_key=config.OPENAI_API_KEY,
)

_translate_llm = ChatOpenAI(
    model=config.TRANSLATION_MODEL,
    temperature=0.2,
    streaming=True,
    api_key=config.OPENAI_API_KEY,
)

_refine_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are correcting a live speech transcript. "
     "Spans marked [REPAIR: ...] contain uncertain or high-risk text (wrong names, mishearings, "
     "garbled numbers, or misheard acronyms). "
     "Fix ONLY the text inside each [REPAIR: ...] marker — replace it with your correction. "
     "Leave all unmarked text exactly as written, character for character. "
     "Return the complete transcript with every [REPAIR: ...] resolved.\n"
     "{context_block}"),
    ("human", "{annotated_transcript}"),
])

_translate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a live interpreter. Translate the text into {target_language}. "
     "Be natural and idiomatic. Do not add explanations or commentary.\n"
     "{glossary_hint}"),
    ("human", "{refined_source}"),
])


def _build_context_block(
    ctx: SessionContext,
    confidence: float | None = None,
    correction_aggressiveness: str = "medium",
) -> str:
    lines = []

    if correction_aggressiveness == "high":
        lines.append(
            "Be aggressive — fix mishearings, wrong names, garbled text, and borderline cases. "
            "Err on the side of correcting over preserving."
        )
    elif correction_aggressiveness == "low":
        lines.append(
            "Be conservative — fix only clear, unambiguous transcription errors. "
            "When uncertain, preserve the original text exactly."
        )
    else:
        # medium: let confidence signal guide the correction intensity
        level = confidence_level(confidence)
        if level == "low":
            lines.append(
                "Transcription confidence is LOW — be thorough: fix likely mishearings, "
                "wrong names, and garbled words."
            )
        elif level == "high":
            lines.append(
                "Transcription confidence is HIGH — make only clear, unambiguous corrections."
            )

    if ctx.get("summary"):
        lines.append(f"Topic: {ctx['summary']}")
    if ctx.get("glossary"):
        pairs = "; ".join(f"{s}={t}" for s, t in list(ctx["glossary"].items())[:25])
        lines.append(f"Known terminology: {pairs}")
    if not lines:
        return ""
    return "Background context (do NOT reproduce in output):\n" + "\n".join(lines)


def _build_glossary_hint(ctx: SessionContext) -> str:
    if not ctx.get("glossary"):
        return ""
    pairs = "; ".join(f"{s}={t}" for s, t in list(ctx["glossary"].items())[:25])
    return f"Known terminology (use these translations consistently): {pairs}"


async def refine_node(state: TranslationState) -> dict:
    raw = state["raw_transcript"]
    ctx = state["session_context"]

    spans = classify_spans(raw, state.get("logprobs"))
    annotated = annotate_transcript(
        raw,
        spans,
        low_threshold=state.get("refinement_low_threshold", config.REFINEMENT_LOW_CONFIDENCE_THRESHOLD),
        high_risk_threshold=state.get("refinement_high_risk_threshold", config.REFINEMENT_HIGH_RISK_THRESHOLD),
    )

    # Nothing flagged — skip the LLM call entirely
    if annotated is None:
        logger.debug("refine_node: all spans protected, skipping LLM")
        return {"refined_source": raw}

    repairable_count = annotated.count("[REPAIR:")
    logger.debug("refine_node: %d span(s) marked for repair", repairable_count)

    context_block = _build_context_block(
        ctx,
        state.get("confidence"),
        state.get("correction_aggressiveness", "medium"),
    )

    chain = _refine_prompt | _refine_llm.with_structured_output(
        RefinementOutput,
        method="function_calling",
    ).with_retry(stop_after_attempt=2)

    result: RefinementOutput = await chain.ainvoke({
        "annotated_transcript": annotated,
        "context_block": context_block,
    })

    refined = result.source.strip() or raw

    # Guard: reject if model leaked topic context into the output
    if len(refined) > len(raw) * 2.5 and len(refined) - len(raw) > 60:
        logger.warning(
            "refine_node: output suspiciously long (raw=%d, refined=%d), using raw",
            len(raw), len(refined),
        )
        refined = raw

    return {"refined_source": refined}


async def translate_node(state: TranslationState) -> dict:
    ctx = state["session_context"]
    glossary_hint = _build_glossary_hint(ctx)

    chain = (_translate_prompt | _translate_llm).with_retry(stop_after_attempt=2)

    result = await chain.ainvoke({
        "refined_source": state["refined_source"],
        "target_language": state["target_language"],
        "glossary_hint": glossary_hint,
    })

    return {"translation": result.content.strip()}


def route_translation(state: TranslationState) -> Literal["translate", "__end__"]:
    if state.get("target_language"):
        return "translate"
    return "__end__"


_builder = StateGraph(TranslationState)
_builder.add_node("refine", refine_node)
_builder.add_node("translate", translate_node)
_builder.add_edge(START, "refine")
_builder.add_conditional_edges("refine", route_translation, {
    "translate": "translate",
    "__end__": END,
})
_builder.add_edge("translate", END)

translation_graph = _builder.compile()
