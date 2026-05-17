# CEO Documentation Assistant

Ask questions about [Collect Earth Online](https://collect-earth-online.org) and get answers drawn directly from the official documentation.

## What you can ask

- How do I create a project in CEO?
- What imagery sources are available?
- How do I set up an institution?
- How does sample collection work?

## How it works

Your question is matched against the CEO documentation using a semantic search index. The most relevant sections are retrieved and passed to Gemini, which writes a concise answer with links back to the source pages.

## Notes

- Answers are only as current as the last time the index was built. If you notice outdated information, ask your team lead to re-run the pipeline.
- The assistant does not have memory across sessions. Each new conversation starts fresh.
