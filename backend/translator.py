import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from . import config

logger = logging.getLogger(__name__)


class TranslationContext:
    """Per-session translation memory: rolling summary + glossary of key terms."""

    def __init__(self):
        self.target_language: str = ""
        self.summary: str = ""
        self.glossary: dict[str, str] = {}            # source_term → translated_term
        self.recent_pairs: list[tuple[str, str]] = [] # (refined_source, translation)
        self.covered_up_to: int = 0
        self._updating: bool = False


@dataclass
class ProcessResult:
    source: str       # refined/corrected source transcription
    translation: str = ""


class Translator:
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def process(
        self,
        raw: str,
        target_language: str,
        ctx: TranslationContext | None = None,
    ) -> ProcessResult:
        """Single GPT call: correct Whisper's transcription + translate.

        If there is no context and no target language, returns the raw text
        unchanged without making an API call.
        """
        if not raw.strip():
            return ProcessResult(source=raw)

        has_context = bool(ctx and (ctx.summary or ctx.glossary))

        # Nothing to do — skip the API call entirely
        if not target_language and not has_context:
            return ProcessResult(source=raw)

        # ── Build prompt ──────────────────────────────────────────────────────

        if target_language:
            task = (
                f"1. Correct: fix any transcription errors in the source text "
                f"(misspelled names, mishearings, wrong technical terms). "
                f"Keep the original language and meaning.\n"
                f"2. Translate: translate the corrected text to {target_language}.\n\n"
                f'Return ONLY this JSON: {{"source": "corrected text", "translation": "translated text"}}'
            )
        else:
            task = (
                "Correct any transcription errors (misspelled names, mishearings, "
                "wrong technical terms). Keep the original language and meaning.\n\n"
                'Return ONLY this JSON: {"source": "corrected text"}'
            )

        messages: list[dict] = [{"role": "system", "content": task}]

        if has_context:
            ctx_lines = []
            if ctx.summary:
                ctx_lines.append(f"Topic: {ctx.summary}")
            if ctx.glossary:
                pairs = "; ".join(f"{s}={t}" for s, t in list(ctx.glossary.items())[:25])
                ctx_lines.append(f"Terminology: {pairs}")
            messages.append({"role": "system", "content": "\n".join(ctx_lines)})

        messages.append({"role": "user", "content": raw})

        resp = await self._client.chat.completions.create(
            model=config.TRANSLATION_MODEL,
            temperature=0.2,
            max_tokens=500,
            messages=messages,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(resp.choices[0].message.content)
            refined = data.get("source", raw).strip() or raw
            translation = data.get("translation", "").strip()
        except (json.JSONDecodeError, AttributeError):
            logger.warning("process(): failed to parse JSON response, using raw")
            return ProcessResult(source=raw)

        if ctx is not None:
            if ctx.target_language and ctx.target_language != target_language:
                # Target language changed — reset memory
                ctx.summary = ""
                ctx.glossary = {}
                ctx.covered_up_to = 0
                ctx.recent_pairs = []
            ctx.target_language = target_language
            ctx.recent_pairs.append((refined, translation))

        return ProcessResult(source=refined, translation=translation)

    async def maybe_update_context(self, ctx: TranslationContext) -> None:
        """Background task: update rolling summary and glossary from recent pairs."""
        UPDATE_EVERY = 5
        if len(ctx.recent_pairs) - ctx.covered_up_to < UPDATE_EVERY or ctx._updating:
            return

        ctx._updating = True
        try:
            new_pairs = ctx.recent_pairs[ctx.covered_up_to:]
            originals = "\n".join(f"- {o}" for o, _ in new_pairs)
            translations = "\n".join(f"- {t}" for _, t in new_pairs)
            current_glossary = json.dumps(ctx.glossary, ensure_ascii=False) if ctx.glossary else "{}"

            prompt = (
                f"You maintain translation memory for a live {ctx.target_language} interpreter.\n\n"
                + (f"Current summary: {ctx.summary}\n\n" if ctx.summary else "")
                + f"Existing glossary (JSON): {current_glossary}\n\n"
                f"Recent source texts:\n{originals}\n\n"
                f"Their {ctx.target_language} translations:\n{translations}\n\n"
                "Return JSON only:\n"
                '{"summary": "...", "glossary": {"source_term": "translation"}}\n\n'
                "Summary: describe the TOPIC in 1-2 sentences "
                "(e.g. 'A political debate about anarchism and the state', "
                "'A cooking tutorial about pasta carbonara'). "
                "Never describe the translation task itself.\n"
                "Glossary: only proper nouns, names, and domain-specific terms "
                "that recur. No common words. Merge with existing entries."
            )

            resp = await self._client.chat.completions.create(
                model=config.TRANSLATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            data = json.loads(resp.choices[0].message.content)
            ctx.summary = data.get("summary", ctx.summary)
            ctx.glossary.update(data.get("glossary", {}))
            ctx.covered_up_to = len(ctx.recent_pairs)
            logger.info(
                "Translation context updated: %d glossary terms, summary=%r",
                len(ctx.glossary), ctx.summary[:80],
            )

        except Exception as exc:
            logger.warning("Translation context update failed: %s", exc)
        finally:
            ctx._updating = False
