import json
import logging

from openai import AsyncOpenAI

from . import config

logger = logging.getLogger(__name__)


class TranslationContext:
    """Per-session translation memory: rolling summary + glossary of key terms."""

    def __init__(self):
        self.target_language: str = ""
        self.summary: str = ""                        # what's being discussed
        self.glossary: dict[str, str] = {}            # source_term → translated_term
        self.recent_pairs: list[tuple[str, str]] = [] # (original, translation) history
        self.covered_up_to: int = 0                   # index into recent_pairs already summarised
        self._updating: bool = False                  # guard against concurrent updates


class Translator:
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def translate(self, text: str, target_language: str, ctx: TranslationContext | None = None) -> str:
        if not text.strip():
            return ""

        messages: list[dict] = [
            # Strict instruction — kept separate so context doesn't dilute it
            {
                "role": "system",
                "content": (
                    f"You are a live interpreter. Translate to {target_language}. "
                    "Reply with the translation and nothing else — no labels, no notes, no commentary."
                ),
            }
        ]

        # Inject context as a separate system message so the model sees it as background
        if ctx and (ctx.summary or ctx.glossary):
            ctx_lines = []
            if ctx.summary:
                ctx_lines.append(f"Topic: {ctx.summary}")
            if ctx.glossary:
                pairs = "; ".join(f"{s}={t}" for s, t in list(ctx.glossary.items())[:25])
                ctx_lines.append(f"Terminology: {pairs}")
            messages.append({"role": "system", "content": "\n".join(ctx_lines)})

        messages.append({"role": "user", "content": text})

        response = await self._client.chat.completions.create(
            model=config.TRANSLATION_MODEL,
            temperature=0.2,
            max_tokens=400,
            messages=messages,
        )

        translation = response.choices[0].message.content.strip()

        # Safety net: strip common model leakage patterns
        for prefix in ("translation:", "tradução:", "traducción:", "traduction:",
                       "übersetzung:", "traduzione:", "перевод:"):
            if translation.lower().startswith(prefix):
                translation = translation[len(prefix):].strip()
                break

        if ctx is not None:
            ctx.recent_pairs.append((text, translation))
            # Reset context if target language changed
            if ctx.target_language and ctx.target_language != target_language:
                ctx.summary = ""
                ctx.glossary = {}
                ctx.covered_up_to = 0
                ctx.recent_pairs = [(text, translation)]
            ctx.target_language = target_language

        return translation

    async def maybe_update_context(self, ctx: TranslationContext) -> None:
        """Background task: update rolling summary and glossary from recent pairs."""
        UPDATE_EVERY = 5
        new_count = len(ctx.recent_pairs) - ctx.covered_up_to

        if new_count < UPDATE_EVERY or ctx._updating:
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
                "Summary rule: describe the TOPIC being discussed in 1-2 sentences "
                "(e.g. 'A football match between Real Madrid and Barcelona', "
                "'A cooking tutorial about pasta carbonara', 'A political debate about immigration'). "
                "Never describe the translation task itself.\n"
                "Glossary rule: only proper nouns, names, and domain-specific technical terms "
                "that appear repeatedly. No common words. Merge with existing entries."
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
                len(ctx.glossary), ctx.summary[:60],
            )

        except Exception as exc:
            logger.warning("Translation context update failed: %s", exc)
        finally:
            ctx._updating = False

    def transcription_hint(self, ctx: TranslationContext | None) -> str:
        """Return a minimal style hint for Whisper — no full sentences or term lists,
        as those get hallucinated as speech continuation."""
        if not ctx or not ctx.summary:
            return ""
        # Just a short domain cue, not a word list — prevents hallucination loops
        return ctx.summary.split(".")[0][:80]
