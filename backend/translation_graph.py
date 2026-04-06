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
    refined_source: str   # populated by refine_node
    translation: str      # populated by translate_node



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
     "You correct live speech transcription errors. Fix mishearings, wrong technical terms, "
     "and misheard proper names. The corrected source must be a faithful correction of what was SAID. "
     "never a description, summary, or paraphrase of the topic. Keep the original language.\n"
     "{context_block}"),
    ("human", "{raw_transcript}"),
])

_translate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a live interpreter. Translate the text into {target_language}. "
     "Be natural and idiomatic. Do not add explanations or commentary.\n"
     "{glossary_hint}"),
    ("human", "{refined_source}"),
])


def _build_context_block(ctx: SessionContext) -> str:
    lines = []
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
    ctx = state["session_context"]
    context_block = _build_context_block(ctx)

    chain = _refine_prompt | _refine_llm.with_structured_output(
        RefinementOutput,
        method="function_calling",
    ).with_retry(stop_after_attempt=2)

    result: RefinementOutput = await chain.ainvoke({
        "raw_transcript": state["raw_transcript"],
        "context_block": context_block,
    })

    refined = result.source.strip() or state["raw_transcript"]

    # Guard: reject if model leaked topic summary into the source text
    raw = state["raw_transcript"]
    if len(refined) > len(raw) * 2.5 and len(refined) - len(raw) > 60:
        logger.warning(
            "refine_node: source suspiciously long (raw=%d, refined=%d), using raw",
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
