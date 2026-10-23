.PHONY: setup dev server web test typecheck check build serve clean sandbox-images test-docker test-opencode

setup:            ## install Python (uv) and UI (npm) dependencies
	uv sync
	cd web && npm ci

dev:              ## agent server + chat room with hot reload; open port 5173
	bash scripts/dev.sh

server:           ## agent server only (port 8000, auto-reload)
	uv run uvicorn baseagent.server.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir src --reload-dir skills

web:              ## chat room only (port 5173)
	cd web && npm run dev

test:             ## Python tests (offline, no API calls)
	uv run pytest -q

sandbox-images:   ## build the code agent's sandbox images (needs Docker)
	docker build -t baseagent-runner:1 docker/runner
	docker build -t baseagent-opencode:1 docker/opencode

test-docker:      ## code-runner tests against real containers (after make sandbox-images)
	uv run pytest -q -m docker tests/test_code_runner.py

test-opencode:    ## the fixer's tools and permissions on a real OpenCode (set OPENCODE_BIN)
	uv run pytest -q -m opencode tests/test_opencode_config.py

typecheck:        ## TypeScript type check for the UI
	cd web && npm run typecheck

check: test typecheck

build:            ## build the UI into web/dist
	cd web && npm run build

serve: build      ## one port: the server also serves the built UI on 8000
	uv run baseagent-server

clean:
	rm -rf web/dist .pytest_cache
