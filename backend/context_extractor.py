"""Background context extraction using Pydantic structured outputs.

Replaces maybe_update_context from the old translator.py.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import config
from .translation_graph import SessionContext

logger = logging.getLogger(__name__)

UPDATE_EVERY = 5

_llm = ChatOpenAI(
    model=config.TRANSLATION_MODEL,
    temperature=0.1,
    api_key=config.OPENAI_API_KEY,
)


class ContextUpdate(BaseModel):
    summary: str = Field(description="1-2 sentence description of the topic being discussed")
    glossary: dict[str, str] = Field(
        default_factory=dict,
        description="Only proper nouns, names, and domain-specific terms that recur. "
                    "Maps source term → translated term. No common words."
    )


_context_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You maintain translation memory for a live interpreter.\n"
     "Analyze the provided transcript pairs and return a JSON object with:\n"
     "- summary: describe the TOPIC in 1-2 sentences (e.g. 'A political debate about anarchism'). "
     "Never describe the translation task itself.\n"
     "- glossary: only recurring proper nouns, names, and technical terms. Merge with existing entries."),
    ("human",
     "{existing_context}"
     "Recent source texts:\n{originals}\n\n"
     "Their {target_language} translations:\n{translations}"),
])

_chain = _context_prompt | _llm.with_structured_output(ContextUpdate, method="function_calling")


async def maybe_update_context(ctx: SessionContext) -> None:
    """Update rolling summary and glossary from recent transcript pairs.

    Mutates ctx in place. Runs as a background task — never awaited on the hot path.
    """
    if len(ctx["recent_pairs"]) - ctx["covered_up_to"] < UPDATE_EVERY:
        return
    if ctx.get("_updating"):
        return

    ctx["_updating"] = True
    try:
        new_pairs = ctx["recent_pairs"][ctx["covered_up_to"]:]
        originals = "\n".join(f"- {o}" for o, _ in new_pairs)
        translations = "\n".join(f"- {t}" for _, t in new_pairs)

        existing_parts = []
        if ctx.get("summary"):
            existing_parts.append(f"Current summary: {ctx['summary']}\n")
        if ctx.get("glossary"):
            import json
            existing_parts.append(
                f"Existing glossary: {json.dumps(ctx['glossary'], ensure_ascii=False)}\n"
            )
        existing_context = "\n".join(existing_parts) + "\n" if existing_parts else ""

        result: ContextUpdate = await _chain.ainvoke({
            "existing_context": existing_context,
            "originals": originals,
            "translations": translations,
            "target_language": ctx.get("target_language", ""),
        })

        ctx["summary"] = result.summary
        ctx["glossary"].update(result.glossary)
        ctx["covered_up_to"] = len(ctx["recent_pairs"])
        logger.info(
            "Translation context updated: %d glossary terms, summary=%r",
            len(ctx["glossary"]), ctx["summary"][:80],
        )

    except Exception as exc:
        logger.warning("Translation context update failed: %s", exc)
    finally:
        ctx["_updating"] = False
