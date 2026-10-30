"""
Eval/agy_judge.py - LangChain LLM wrapper around any agent CLI that has a
non-interactive print mode, so it can be used as the Ragas judge.

Presets: agy, openclaude, claude, gemini
Optional env vars:
    JUDGE_MODEL      -> passed as --model (not used for gemini preset)
    JUDGE_PROVIDER   -> passed as --provider (openclaude only)
    JUDGE_TIMEOUT    -> seconds per call (default 180)
"""

import os
import shutil
import subprocess
import tempfile
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM

TIMEOUT = int(os.getenv("JUDGE_TIMEOUT", os.getenv("AGY_TIMEOUT", "180")))
MODEL = os.getenv("JUDGE_MODEL", os.getenv("AGY_MODEL", ""))
PROVIDER = os.getenv("JUDGE_PROVIDER", "")

# Run from an empty scratch dir so the agent can't wander through the project.
_SCRATCH = tempfile.mkdtemp(prefix="cli_judge_")

_PREFIX = (
    "You are a strict evaluation judge. Do NOT use tools, read files or run commands. "
    "Answer ONLY with what the prompt asks for. If the prompt asks for JSON, output raw "
    "valid JSON only, with no markdown fences and no commentary.\n\n"
)


def _build_cmd(cli: str, prompt: str, model: str, provider: str, timeout: int) -> list[str]:
    binary = shutil.which(cli) or os.path.expanduser(f"~/.local/bin/{cli}")
    if cli == "agy":
        cmd = [binary, "-p", prompt, "--print-timeout", f"{timeout}s"]
        if model:
            cmd += ["--model", model]
    elif cli == "openclaude":
        cmd = [binary, "-p", prompt, "--no-session-persistence"]
        if provider:
            cmd += ["--provider", provider]
        if model:
            cmd += ["--model", model]
    elif cli == "claude":
        cmd = [binary, "-p", prompt, "--no-session-persistence"]
        if model:
            cmd += ["--model", model]
    elif cli == "opencode":
        # JUDGE_MODEL must be "provider/model" (see `opencode models`)
        cmd = [binary, "run"]
        if model:
            cmd += ["-m", model]
        cmd += [prompt]
    elif cli == "gemini":
        cmd = [binary, "-p", prompt]
        if model:
            cmd += ["-m", model]
    else:
        raise ValueError(f"Unknown judge CLI: {cli}")
    return cmd


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text.strip("`")
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text


class CliJudgeLLM(LLM):
    cli: str = "agy"
    model: str = MODEL
    provider: str = PROVIDER
    timeout: int = TIMEOUT

    @property
    def _llm_type(self) -> str:
        return f"{self.cli}-cli"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        cmd = _build_cmd(self.cli, _PREFIX + prompt, self.model, self.provider, self.timeout)
        # stdin=DEVNULL avoids non-TTY hangs in some CLIs
        proc = subprocess.run(
            cmd,
            cwd=_SCRATCH,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self.timeout + 30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.cli} exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:500]}"
            )
        return _strip_fences(proc.stdout)


# backwards-compatible name
AntigravityLLM = CliJudgeLLM