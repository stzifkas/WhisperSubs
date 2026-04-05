from openai import AsyncOpenAI
from . import config


class Translator:
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def translate(self, text: str, target_language: str) -> str:
        if not text.strip():
            return ""

        response = await self._client.chat.completions.create(
            model=config.TRANSLATION_MODEL,
            temperature=0,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate the following text to {target_language}. "
                        "Output only the translation, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )

        return response.choices[0].message.content.strip()
