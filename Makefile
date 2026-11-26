.PHONY: setup knowledge colab-setup colab-login colab-check dev start stop dev-log server web test typecheck check build serve clean sandbox-images test-docker test-opencode

setup:            ## install Python (uv) and UI (npm) dependencies
	uv sync
	cd web && npm ci

knowledge:        ## pull the latest merged skills into knowledge/ (new chats use them; commit the moved pointer)
	git submodule update --init --remote knowledge
	@echo "knowledge/ is at $$(git -C knowledge log --oneline -1)"

colab-setup:      ## install the isolated Google Colab CLI for T4 notebook runs
	uv tool install google-colab-cli==0.8.0

COLAB_SCOPES = openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

colab-login:      ## sign in to Google once for Colab T4 runs (kept across rebuilds)
	gcloud auth application-default login --no-launch-browser --scopes=$(COLAB_SCOPES)
	colab --auth adc usage

colab-check:      ## confirm the Colab CLI can use your Google sign-in
	colab --auth adc usage

dev:              ## agent server + chat room with hot reload in this terminal; open port 5173
	bash scripts/dev.sh

start:            ## the same, in the background (the dev container does this on every start/resume)
	bash scripts/background.sh start

stop:             ## stop the background chat room
	bash scripts/background.sh stop

dev-log:          ## follow the background chat room's log
	@touch runs/dev-server.log 2>/dev/null || (mkdir -p runs && touch runs/dev-server.log)
	tail -n 50 -f runs/dev-server.log

server:           ## agent server only (port 8000, auto-reload)
	uv run uvicorn baseagent.server.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir src --reload-dir skills

web:              ## chat room only (port 5173)
	cd web && npm run dev

test:             ## Python tests (offline, no API calls)
	uv run pytest -q

.PHONY: eval-card
eval-card:        ## grade the newest card-CNN Colab run (or NOTEBOOK=workspace/<chat>/outputs/baseagent-*/analysis.ipynb)
	uv run python evals/card_cnn.py $(NOTEBOOK)

sandbox-images:   ## build the code agent's sandbox images (needs Docker)
	docker build -t baseagent-runner:1 docker/runner
	docker build -t baseagent-opencode:1 docker/opencode

test-docker:      ## runner + full code-fix pipeline in real containers (after make sandbox-images)
	uv run pytest -q -m docker tests/test_code_runner.py tests/test_code_fixer.py

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
