"""Active glossary enforcement.

After translation, scans the output for glossary violations — cases where a
source term that is present in the segment was not translated using the
glossary-mandated target term.  Detected violations trigger a targeted repair
call that fixes only the offending term(s), leaving the rest of the translation
untouched.

Metrics (hits / misses / violations / repaired) are emitted as structured INFO
log lines so they can be parsed by log aggregators.
"""
import logging
import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from . import config

logger = logging.getLogger(__name__)



@dataclass
class GlossaryViolation:
    """A single term that was present in the source but mis-translated."""
    source_term: str       # term that appeared in the source segment
    expected_target: str   # glossary-mandated translation


@dataclass
class GlossaryCheckResult:
    """Outcome of a glossary consistency check for one translation segment."""
    hits: int                                      # terms correctly translated
    misses: int                                    # terms not in this segment
    violations: list[GlossaryViolation] = field(default_factory=list)
    repaired: bool = False
    translation: str = ""                          # final (possibly repaired) translation



def _term_present(term: str, text: str) -> bool:
    """Return True if *term* occurs (case-insensitively) in *text*.

    Uses a simple regex search without strict word-boundary anchors so that
    the check works correctly for non-Latin scripts and compound words.
    """
    return bool(re.search(re.escape(term), text, re.IGNORECASE))


def detect_violations(
    source: str,
    translation: str,
    glossary: dict[str, str],
) -> tuple[list[GlossaryViolation], int, int]:
    """Scan one translated segment for glossary consistency violations.

    Returns:
        violations: terms that were in the source but wrongly translated
        hits:       count of terms correctly translated
        misses:     count of glossary terms absent from this source segment
    """
    violations: list[GlossaryViolation] = []
    hits = 0
    misses = 0

    for source_term, target_term in glossary.items():
        if not _term_present(source_term, source):
            misses += 1
            continue
        if _term_present(target_term, translation):
            hits += 1
        else:
            violations.append(GlossaryViolation(
                source_term=source_term,
                expected_target=target_term,
            ))

    return violations, hits, misses



def log_metrics(
    violations: list[GlossaryViolation],
    hits: int,
    misses: int,
    repaired: bool,
    session_id: str = "",
) -> None:
    """Emit a structured INFO line for log aggregation."""
    violated_terms = [v.source_term for v in violations]
    logger.info(
        "Glossary check session=%s hits=%d misses=%d violations=%d repaired=%s%s",
        session_id or "-",
        hits,
        misses,
        len(violations),
        repaired,
        f" violated={violated_terms}" if violated_terms else "",
    )



_repair_llm = ChatOpenAI(
    model=config.TRANSLATION_MODEL,
    temperature=0.1,
    streaming=False,
    api_key=config.OPENAI_API_KEY,
)

_repair_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are correcting a translation for consistent terminology.\n"
     "Apply ONLY the following corrections — change nothing else in the translation:\n"
     "{corrections}\n"
     "Return the corrected translation only, with no explanation or commentary."),
    ("human", "{translation}"),
])


async def repair_glossary_violations(
    translation: str,
    violations: list[GlossaryViolation],
) -> str:
    """Return a translation with the violated terms replaced by their glossary targets.

    Only the listed terms are touched; all other text is preserved verbatim.
    Falls back to the original translation if the LLM call fails.
    """
    if not violations:
        return translation

    corrections = "\n".join(
        f'- Wherever "{v.source_term}" was translated, replace that translation '
        f'with "{v.expected_target}"'
        for v in violations
    )
    chain = (_repair_prompt | _repair_llm).with_retry(stop_after_attempt=2)
    try:
        result = await chain.ainvoke({
            "translation": translation,
            "corrections": corrections,
        })
        return result.content.strip() or translation
    except Exception as exc:
        logger.warning("Glossary repair LLM call failed: %s", exc)
        return translation


async def check_and_repair(
    source: str,
    translation: str,
    glossary: dict[str, str],
    session_id: str = "",
) -> GlossaryCheckResult:
    """Run the full check → (optional repair) → metrics pipeline.

    Returns a `GlossaryCheckResult` whose `.translation` field is the
    authoritative final translation (repaired if violations were found).
    """
    if not glossary or not translation:
        return GlossaryCheckResult(hits=0, misses=0, translation=translation)

    violations, hits, misses = detect_violations(source, translation, glossary)
    repaired = False
    final_translation = translation

    if violations:
        final_translation = await repair_glossary_violations(translation, violations)
        repaired = True

    log_metrics(violations, hits, misses, repaired, session_id)

    return GlossaryCheckResult(
        hits=hits,
        misses=misses,
        violations=violations,
        repaired=repaired,
        translation=final_translation,
    )
