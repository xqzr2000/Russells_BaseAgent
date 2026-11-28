# Run it in GitHub Codespaces

## 1. Create a Codespace

Fork or clone the repository, then create a Codespace on it. The dev container installs
Python 3.12, Node.js 22, `uv`, Stockfish and Docker, then runs `uv sync` and `npm ci`.

`uv sync` also installs the data science stack the data scientist's kernel uses (pandas,
numpy, scipy, matplotlib, seaborn, scikit-learn, statsmodels, pyarrow, openpyxl, and CPU
builds of torch and torchvision for smoke runs). In an existing Codespace, run `uv sync`
once after pulling.

## 2. Add `OPENAI_API_KEY` as a Codespaces secret

- In GitHub, go to **Settings → Codespaces → New secret**.
- Name it `OPENAI_API_KEY` and grant access to this repository.
- Optional: set `OPENAI_MODEL` to override the default model (`gpt-5-mini`).
- Without a key, the chat room still works with the offline `fake-echo` model.

## 3. Open the chat room

- The Codespace starts the chat room by itself every time it starts or resumes. Open the
  **Chat room** port (5173) from the Ports tab.
- `make dev-log` follows its log, `make stop` stops it and `make start` starts it again.
- `make dev` runs it in your terminal instead (it stops the background copy first). Set
  `BASEAGENT_AUTOSTART=0` as a Codespaces variable to turn auto-start off.
- Chats live in the agent server's memory, so they don't survive a restart. A tab left
  open from before tells you so and starts a new chat with your next message.

## Useful commands

| Command | What it does |
| --- | --- |
| `make test` | Offline Python tests (no API calls) |
| `make typecheck` | TypeScript check for the UI |
| `make knowledge` | Pull the latest private skills into `knowledge/` |
| `make eval-card` | Grade the newest card-classifier Colab run |
| `make sandbox-images` / `make test-docker` | Build the code runner's image, and test it in real containers |
| `make serve` | One port: the server also serves the built UI on 8000 |

For Colab T4 runs, see [Colab setup](colab.md).
