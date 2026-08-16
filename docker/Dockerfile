# A flowcoder runner image: the `claude` CLI, the engine, and nothing task-specific.
#
# Everything a run needs arrives on mounted volumes — flowchart JSON at
# /work/commands, the project Claude works on at /work/workspace — so the image
# stays static across a whole testbench sweep and only the mounts and the model
# endpoint vary.
#
#   docker build -t flowcoder -f docker/Dockerfile .
#   docker run --rm \
#     -e ANTHROPIC_BASE_URL=http://vllm:8000 \
#     -e ANTHROPIC_AUTH_TOKEN=local \
#     -e ANTHROPIC_MODEL=my-local-model \
#     -v ./commands:/work/commands:ro \
#     -v ./workspace:/work/workspace \
#     flowcoder ex0-design-doc "a CSV to JSON converter"
#
# See docker/compose.yaml for the same thing wired to a vLLM service.

FROM python:3.12-slim-bookworm

# `latest`, `stable`, or an exact version (e.g. 2.1.233).  Pin it for a
# testbench: a CLI upgrade mid-sweep makes runs incomparable.
ARG CLAUDE_VERSION=latest

# curl for the installer; git and ripgrep because flowchart bash blocks and
# Claude's own search tools reach for them constantly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Claude Code ships as a native binary installed under $HOME/.local.  The
# installer takes stable|latest|VERSION as its only argument.
RUN curl -fsSL https://claude.ai/install.sh | bash -s "${CLAUDE_VERSION}"
ENV PATH="/root/.local/bin:${PATH}"

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency layer first: it changes only when the lockfile does, so editing
# engine source does not re-resolve the world.
COPY pyproject.toml uv.lock ./
COPY packages ./packages

# --frozen fails loudly if uv.lock is stale rather than silently resolving
# something else.  The dev extra brings pytest in, so the same image can run the
# suite against the model it is pointed at (see tests/local_model).
RUN uv sync --frozen --extra dev

ENV PATH="/app/.venv/bin:${PATH}"

# The two mount points.  Created here so a run with no mounts still starts and
# fails with "command not found" rather than a confusing path error.
RUN mkdir -p /work/commands /work/workspace

COPY docker/entrypoint.sh /usr/local/bin/flowcoder-entrypoint
RUN chmod +x /usr/local/bin/flowcoder-entrypoint

WORKDIR /work/workspace

ENTRYPOINT ["/usr/local/bin/flowcoder-entrypoint"]
CMD ["--list"]
