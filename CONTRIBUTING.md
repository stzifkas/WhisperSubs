# Contributing to WhisperSubs

Thanks for your interest in contributing! Here's everything you need to get started.

---

## Ways to contribute

- **Bug reports** — open an issue with steps to reproduce, expected vs. actual behaviour, and relevant log output
- **Feature requests** — open an issue describing the use case; the more concrete the better
- **Code** — bug fixes, new features, performance improvements, refactors
- **Docs** — corrections, clearer explanations, examples

---

## Development setup

```bash
git clone https://github.com/stzifkas/whispersubs.git
cd whispersubs

uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

uvicorn backend.main:app --reload
```

Or with Docker:

```bash
docker compose up --build
```

### Running tests

```bash
pytest
```

The suite is fast and runs without network access or an API key (model calls are mocked). Keep it green, and add tests for any behaviour change.

---

## Claiming an issue

Before writing code, **comment on the issue to claim it** (e.g. "Taking this one"). This prevents two people from building the same fix, and it's also how a maintainer can add you as the assignee — GitHub only lets us assign people who have commented or are collaborators.

First check the issue isn't already claimed or linked to an open PR. Please work one issue at a time; it keeps things fair for other contributors.

---

## Submitting a pull request

1. Fork the repo and create a branch from `main`:
   ```bash
   git checkout -b fix/my-bug-fix
   ```

2. Make your changes. Keep commits focused — one logical change per commit, and **one issue per pull request**.

3. Run the test suite and add tests for any behaviour change:
   ```bash
   pytest
   ```
   Also verify manually where it helps: start the server, capture some audio, and confirm captions and translation work end-to-end.

4. Open a PR against `main`. Link the issue with `Closes #123` in the description, and include:
   - What the change does and why
   - How you tested it (and that `pytest` passes)
   - Screenshots or log excerpts if relevant

---

## Code style

- Python: follow PEP 8; type-annotate new functions
- JavaScript: plain ES2020+, no build step, no frameworks
- Keep dependencies minimal — think carefully before adding a new package

---

## Project structure

```
backend/
  main.py              # FastAPI app, WebSocket handler, chat endpoint
  translation_graph.py # LangGraph pipeline (refine → translate)
  translator.py        # process_and_stream() — streams graph events to WebSocket
  context_extractor.py # background: rolling summary + glossary
  vector_store.py      # per-session in-memory embeddings for chat retrieval
  config.py            # env-based configuration
frontend/
  index.html
  app.js               # audio capture, WebSocket, caption display
  style.css
```

---

## Reporting issues

Please include:
- Browser and OS
- Server logs (run with `--log-level debug` for more detail)
- Whether you're using tab capture or microphone
- The source and target languages

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
