"""CLI argument parsing for the flowcoder-engine binary.

Flowcoder-engine is a transparent proxy for claude CLI.  It parses its own
flags and common Claude settings, then passes everything else through to the
inner claude subprocess.

Standalone usage (SDK users):
    flowcoder-engine --search-path ./commands --model sonnet

Embedded usage (Axi / host frameworks):
    flowcoder-engine --search-path ./commands --output-format stream-json ...
    (host passes all Claude flags via passthrough)
"""

from __future__ import annotations

import argparse
import shlex
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flowcoder_flowchart import Argument

SDK_VERSION = "0.1.39"


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags that configure the engine itself (not inner Claude).

    Shared by the ``flowcoder-engine`` proxy and the ``flowcoder`` runner so
    both accept the same command-resolution flags.
    """
    parser.add_argument(
        "--claude-path",
        help="Path to the claude CLI binary (auto-detected if not specified)",
    )
    parser.add_argument(
        "--search-path",
        action="append",
        dest="search_paths",
        help="Search paths for flowchart command resolution (can specify multiple)",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=1000,
        help=(
            "Maximum number of blocks to execute per flowchart (safety limit); "
            "0 or less disables it, for a deliberately unbounded flowchart"
        ),
    )


def add_claude_args(parser: argparse.ArgumentParser) -> None:
    """Add the Claude settings that ``build_inner_claude_cmd`` consumes.

    These provide a clean interface for callers who don't want to know raw
    Claude CLI flags.  All are optional.  Every flag added here is read by
    ``build_inner_claude_cmd``/``build_inner_env``, so any parser that feeds
    those functions must call this to have the attributes defined.
    """
    parser.add_argument(
        "--model",
        help="Model to use for the inner Claude process (e.g. sonnet, opus, haiku)",
    )
    parser.add_argument(
        "--permission-mode",
        help="Permission mode for the inner Claude process (default, plan, bypassPermissions)",
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt for the inner Claude process",
    )
    parser.add_argument(
        "--append-system-prompt",
        help="Append to the default system prompt",
    )
    parser.add_argument(
        "--mcp-config",
        dest="mcp_config",
        help="MCP server config (JSON string or file path)",
    )
    parser.add_argument(
        "--resume",
        help="Resume a previous Claude session by ID",
    )
    parser.add_argument(
        "--cwd",
        help="Working directory for the inner Claude process",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Anthropic API key (alternative to ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of agentic turns",
    )
    parser.add_argument(
        "--disallowed-tools",
        dest="disallowed_tools",
        help="Comma-separated list of tools to disallow",
    )
    parser.add_argument(
        "--allowed-tools",
        dest="allowed_tools",
        help="Comma-separated list of tools to allow",
    )
    parser.add_argument(
        "--json-schema",
        dest="json_schema",
        help=(
            "JSON Schema for the flowchart's FINAL answer. Parsed here and "
            "applied once, after the flowchart completes -- deliberately NOT "
            "forwarded to the inner CLI, where it would constrain every turn "
            "and override each block's own output_schema"
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Engine-specific flags and common Claude settings are parsed explicitly.
    All remaining arguments are collected in args.passthrough and forwarded
    to the inner claude process as-is.
    """
    parser = argparse.ArgumentParser(
        prog="flowcoder-engine",
        description="Claude CLI proxy with flowchart execution",
    )

    add_engine_args(parser)
    add_claude_args(parser)

    # Accepted but inert: the inner CLI is always run with --verbose,
    # because -p plus --output-format stream-json requires it (see
    # build_inner_claude_cmd).  The flag is kept so that a host passing
    # --verbose has it absorbed here rather than duplicated through
    # passthrough, and so existing callers keep working.
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Accepted for compatibility; the inner Claude CLI is always "
             "run with --verbose, which the stream-json protocol requires",
    )

    args, remaining = parser.parse_known_args(argv)
    args.passthrough = remaining
    return args


def build_inner_claude_cmd(args: argparse.Namespace, claude_path: str) -> list[str]:
    """Build the inner Claude CLI command from parsed args.

    Combines engine-owned flags, explicit Claude settings, and raw
    passthrough args into a single command list.

    Priority (highest to lowest):
    1. Passthrough args (raw Claude CLI flags from the host)
    2. Engine-owned Claude settings (--model, --permission-mode, etc.)
    3. Defaults (stream-json I/O, -p mode)
    """
    # --verbose is NOT optional here, despite the flag of the same name.
    # Claude CLI rejects `-p` combined with `--output-format stream-json`
    # unless --verbose is also present, exiting immediately with
    #   "When using --print, --output-format=stream-json requires --verbose"
    # An inner CLI that exits at once gives EOF on stdout, so read() returns
    # None and every query comes back empty — a silent wrong answer rather
    # than a crash. Since -p and stream-json are hardcoded above, the
    # requirement always applies, so this is emitted unconditionally.
    #
    # Do NOT re-gate this on args.verbose or move it behind a condition.
    # Hosts that pass their own --verbose have it consumed by our parser
    # (it never reaches passthrough), so they depend on this line to put it
    # back; re-gating breaks them silently with the instant-exit signature
    # above and no test catches it downstream.
    cmd = [
        claude_path,
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]

    if args.model:
        cmd.extend(["--model", args.model])

    if args.permission_mode:
        cmd.extend(["--permission-mode", args.permission_mode])

    if args.system_prompt:
        cmd.extend(["--system-prompt", args.system_prompt])

    if args.append_system_prompt:
        cmd.extend(["--append-system-prompt", args.append_system_prompt])

    if args.mcp_config:
        cmd.extend(["--mcp-config", args.mcp_config])

    if args.resume:
        cmd.extend(["--resume", args.resume])

    if args.max_turns:
        cmd.extend(["--max-turns", str(args.max_turns)])

    if args.disallowed_tools:
        cmd.extend(["--disallowedTools", args.disallowed_tools])

    if args.allowed_tools:
        cmd.extend(["--allowedTools", args.allowed_tools])

    # Passthrough: raw Claude CLI flags from the host framework.
    # These override anything above if there are conflicts (Claude CLI
    # uses last-wins semantics for most flags).
    if args.passthrough:
        cmd.extend(args.passthrough)

    return cmd


def build_inner_env(args: argparse.Namespace) -> dict[str, str]:
    """Build environment for the inner Claude CLI process.

    Sets the env vars that make inner Claude use the SDK control protocol
    (for tool permissions and MCP), without depending on the full
    claude-code-sdk package.

    Key vars (copied from claude-code-sdk's SubprocessCLITransport.connect):
    - CLAUDE_CODE_ENTRYPOINT: tells Claude CLI this is an SDK session
    - CLAUDE_AGENT_SDK_VERSION: version identifier for the SDK protocol
    - Strips CLAUDECODE: prevents nested-session rejection
    """
    import os

    env = dict(os.environ)

    # Remove nested-session guard — the engine IS the outer session
    env.pop("CLAUDECODE", None)

    # Tell inner Claude to use the SDK control protocol.
    # Without these, Claude CLI auto-denies tool permissions in pipe mode.
    env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
    env["CLAUDE_AGENT_SDK_VERSION"] = SDK_VERSION

    # API key override
    if args.api_key:
        env["ANTHROPIC_API_KEY"] = args.api_key

    # CWD override
    if args.cwd:
        env["PWD"] = args.cwd

    return env


def build_variables(
    args_string: str,
    declared_args: list[Argument],
) -> dict[str, Any]:
    """Build initial variable dict from a shell-style args string.

    Maps positional args to $1, $2, etc. and to declared argument names.
    """
    if args_string.strip():
        parts = shlex.split(args_string)
    else:
        parts = []

    variables: dict[str, Any] = {}

    for i, arg_def in enumerate(declared_args):
        pos = i + 1
        if i < len(parts):
            variables[f"${pos}"] = parts[i]
            variables[arg_def.name] = parts[i]
        elif arg_def.default is not None:
            variables[f"${pos}"] = arg_def.default
            variables[arg_def.name] = arg_def.default
        elif arg_def.required:
            raise ValueError(
                f"Missing required argument: {arg_def.name} (position {pos})"
            )

    # Extra positional beyond declared
    for i in range(len(declared_args), len(parts)):
        variables[f"${i + 1}"] = parts[i]

    return variables
