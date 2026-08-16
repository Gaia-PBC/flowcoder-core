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

A `spawn` block may set `search_path` to aim one spawn at a specific flowchart —
a bundle's own, say, which the parent's search paths need not cover. That path
is searched *before* step 1, so a same-named file under the working directory or
in a `--search-path` cannot silently capture the spawn. The spawned child
inherits it at the same precedence, so its own `command` blocks resolve there
too. Templates (`{{var}}`, `$1`) are substituted first.

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

## Tests

```bash
uv run pytest packages/flowcoder-engine/tests/unit   # engine unit tests
uv run pytest packages/flowcoder-flowchart/tests     # data model tests
uv run pytest tests -m "not slow"                    # cross-package tests
```

`tests/` spans both packages. Tests marked `slow` drive a real `claude` CLI
and cost tokens; deselect them with `-m "not slow"`.

## License

MIT — see [LICENSE](LICENSE).
