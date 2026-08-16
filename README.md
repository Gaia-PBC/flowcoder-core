# FlowCoder Engine

Execute flowchart-defined workflows against Claude Code.

The engine is a transparent proxy around the `claude` CLI: it behaves
identically to `claude -p --input-format stream-json --output-format
stream-json`, forwarding stdin and stdout to an inner Claude process. When a
user message contains a slash command matching a known flowchart, the engine
takes over, runs the flowchart, emits structured events, then resumes proxying.

The tkinter GUI that used to live here now has its own repo:
[px-pride/flowcoder-tk-gui](https://github.com/px-pride/flowcoder-tk-gui).

## Packages

| Package | Purpose |
|---|---|
| `packages/flowcoder-flowchart` | Pure pydantic data models — blocks, connections, commands. No I/O. |
| `packages/flowcoder-engine` | Execution engine, CLI proxy, stream-json protocol, graph walker, terminal runner. |

`flowcoder-engine` depends on `flowcoder-flowchart`. Nothing depends on the GUI.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- The `claude` CLI on `PATH` (or pass `--claude-path`)

## Install

```bash
uv sync --extra dev
```

The repo root is not itself an installable package — it pins the two packages
as editable local sources and hosts the cross-package tests in `tests/`.

To depend on the engine from another project:

```toml
[tool.uv.sources]
flowcoder-engine = { git = "https://github.com/px-pride/flowcoder.git", subdirectory = "packages/flowcoder-engine" }
flowcoder-flowchart = { git = "https://github.com/px-pride/flowcoder.git", subdirectory = "packages/flowcoder-flowchart" }
```

Pin a `rev` for reproducible builds.

## Usage

Two entry points ship with `flowcoder-engine`. Use `flowcoder` to run a
flowchart yourself; use `flowcoder-engine` to embed the engine in a host
framework.

### `flowcoder` — run a command in the terminal

```bash
uv run flowcoder <command> [arguments...]
```

```bash
uv run flowcoder --list                                  # what can I run?
uv run flowcoder ex0-design-doc "a CSV to JSON converter"
uv run flowcoder --search-path ./commands --model sonnet mycommand "an argument"
uv run flowcoder --json mycommand > result.json          # final variables, machine-readable
```

Block progress and the agent's replies stream to the terminal as the flowchart
runs; `input` blocks read a line from stdin. The exit code is 0 when the
flowchart completes, the block's own code when an `exit` block ends the run, and
1 otherwise.

Flags go **before** the command name — everything after it is passed to the
flowchart as `$1`, `$2`, … so a flowchart argument that looks like a flag stays
an argument.

| Flag | Purpose |
|---|---|
| `--list`, `-l` | List resolvable commands and exit |
| `--json` | Print the final variables as JSON on stdout (progress moves to stderr) |
| `--verbose`, `-v` | Show engine logs, per-block completions and raw events |
| `--yes`, `-y` | Auto-allow tool permission requests instead of asking |
| `--no-color` | Disable ANSI colour |

Sessions run with `--permission-mode bypassPermissions` by default, since
flowcharts are meant to run unattended. Pass `--permission-mode default` (or
`plan`) and each tool call that needs approval is asked on the terminal —
`[y]es`, `[n]o`, or `[a]lways` to stop asking for that tool. `--yes` allows
everything without asking.

> Only `flowcoder` prompts. The `flowcoder-engine` proxy leaves permission
> handling to its host, and Claude denies un-approved tools when neither is
> wired up.

### `flowcoder-engine` — the stream-json proxy

```bash
uv run flowcoder-engine [options]
```

This is the embeddable entry point: it reads JSON messages on stdin and writes
them on stdout, so it expects a host framework on the other end rather than a
person.

### Shared flags

Both accept the same engine and Claude settings; run either with `--help` for
the full list. Commonly used:

| Flag | Purpose |
|---|---|
| `--claude-path` | Path to the `claude` binary (auto-detected if omitted) |
| `--search-path` | Extra directory to resolve flowchart commands from (repeatable) |
| `--max-blocks` | Safety limit on blocks executed per flowchart |
| `--model` | Model for the inner Claude process (e.g. `sonnet`, `opus`, `haiku`) |
| `--permission-mode` | `default`, `plan`, or `bypassPermissions` |
| `--cwd` | Working directory for the inner Claude process |
| `--resume` | Resume a previous Claude session by ID |

## Command resolution

A command `name` (written `/name` in a proxied message, and either way on the
`flowcoder` command line) resolves to `name.json`, searched in this order
(`resolver.py`):

1. `./commands/name.json`, then `./name.json`
2. For each `--search-path` *P*: `P/name.json`, then `P/commands/name.json`
3. `~/.flowcoder/commands/name.json`

First match wins; `CommandNotFoundError` if none.

## Flowchart format

A command is a JSON document of blocks and the connections between them.

### Block types

| Block | Purpose |
|---|---|
| `start` | Entry point (required) |
| `end` | Exit point |
| `prompt` | Send a prompt to Claude, capture structured output |
| `bash` | Execute a shell command |
| `variable` | Set a variable, with type coercion |
| `branch` | Conditional branching on a variable |
| `command` | Invoke another command |
| `refresh` | Restart the Claude session |
| `spawn` | Start a named background agent |
| `wait` | Block until spawned agents finish |
| `exit` | Terminate the flowchart early |
| `input` | Request input mid-run |

Variable types: `string`, `number`, `boolean`, `json` (`int` and `float` are
accepted as aliases for `number`).

### Branch conditions

Templates are substituted before the condition is evaluated. If both sides
parse as numbers the comparison is numeric, otherwise it is a string compare.

| Form | Example |
|---|---|
| `field` (truthy) | `isComplete` |
| `!field` (negated) | `!hasErrors` |
| `field == value` | `status == "done"` |
| `field != value` | `count != 0` |
| `field > value` | `score > 80` |
| `field < value` | `attempts < 3` |
| `field >= value` | `progress >= 100` |
| `field <= value` | `errors <= 5` |

### Variable substitution

- `$1`, `$2`, … — positional arguments passed to the command
- `{{variable_name}}` — variables set by earlier blocks

## Example commands

`commands/` holds runnable examples:

| Command | Behaviour |
|---|---|
| `/ex0-design-doc "<task>"` | Write a design doc for the given task |
| `/ex1-do-until-done <doc>` | Implement a design doc, then audit and loop |
| `/ex2-testing-loop <doc>` | Write a test suite, then loop fixing failures |
| `/ex3-improve-project <N>` | N rounds of designing and building a feature |
| `/all-examples` | Run the four above in sequence |

## Local models

The engine needs no code change to target a locally served model. The `claude`
CLI it proxies reads its endpoint from the environment, and the engine passes
its own environment through:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=my-local-model \
uv run flowcoder --search-path ./commands mycommand "some text"
```

The one hard constraint is the wire format: `claude` speaks the Anthropic
Messages API and nothing else, so the endpoint must serve `/v1/messages`. vLLM
does. An OpenAI-only server needs a translating proxy in front.

`ANTHROPIC_BASE_URL` takes no trailing `/v1` — the CLI appends `/v1/messages`
itself.

## Docker

`docker/Dockerfile` builds a runner image: the `claude` CLI, the engine, and
nothing task-specific. Flowchart JSON and the project being worked on arrive on
mounts, so one image serves a whole sweep.

```bash
docker build -t flowcoder -f docker/Dockerfile .

docker run --rm \
  -e ANTHROPIC_BASE_URL=http://vllm:8000 \
  -e ANTHROPIC_AUTH_TOKEN=local \
  -e ANTHROPIC_MODEL=my-local-model \
  -v ./my-flowcharts:/work/commands:ro \
  -v ./workspace:/work/workspace \
  flowcoder soul "build me a CSV parser"
```

The command name is the mounted file's stem — `soul.json` runs as `soul`.
Output is `--json` by default, and the exit code is the flowchart's, so a
harness can consume both directly.

| Mount / variable | Purpose |
|---|---|
| `/work/commands` | Flowchart JSON. Overridable with `FLOWCODER_SEARCH_PATH`. |
| `/work/workspace` | Where Claude does its work. Overridable with `FLOWCODER_CWD`. |
| `FLOWCODER_JSON` | `0` for human-readable output instead of JSON. |
| `CLAUDE_VERSION` | Build arg. Pin it for a sweep — a CLI upgrade partway makes runs incomparable. |

The entrypoint logs the endpoint in use on stderr, so a run that silently fell
back to the billed API is visible in the logs rather than on the invoice.

`docker/compose.yaml` wires the same image to a vLLM service.

## Tests

```bash
uv run pytest packages/flowcoder-engine/tests/unit   # engine unit tests
uv run pytest packages/flowcoder-flowchart/tests     # data model tests
uv run pytest tests -m "not slow"                    # cross-package tests
```

Three tiers, by what they talk to:

| Tier | Talks to | When it runs |
|---|---|---|
| default | `_stub_claude.py`, a zero-token stub | always |
| `local_model/` | a locally served model | when `ANTHROPIC_BASE_URL` is set |
| `-m slow` | the real `claude` CLI, costs tokens | when selected |

The local-model tier asks whether a given model can actually drive a flowchart —
does `output_schema` come back as parseable JSON, does a branch follow the
answer. Run it with:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_MODEL=my-local-model \
uv run pytest packages/flowcoder-engine/tests/local_model
```

The gate is `ANTHROPIC_BASE_URL` — the same variable that points the CLI at the
endpoint — so these tests cannot run against the billed API by construction.
`ANTHROPIC_MODEL` is optional; without it the CLI's default model applies.
`FLOWCODER_LOCAL_MODEL_TIMEOUT` bounds each run (default 300s).

Note that the excluded tiers **deselect** rather than skip: `tests/conftest.py`
turns a skipped test into a failure, on the grounds that a skip is evidence of
nothing.

## License

MIT — see [LICENSE](LICENSE).
