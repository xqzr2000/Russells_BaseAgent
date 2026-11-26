"""Execute a complete notebook in a short-lived Google Colab T4 session."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NotRequired, TypedDict
from uuid import uuid4

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

from baseagent.events import Attachment
from baseagent.sandbox.kaggle import kaggle_dataset, kaggle_download_source
from baseagent.sandbox.kernel import CellResult, KernelSandbox
from baseagent.sandbox.probe import DOWNLOAD_TIMEOUT, ensure_dataset
from baseagent.sandbox.notebook import NOTEBOOK_MIME, SETUP_CODE, NotebookRecorder
from baseagent.tools import ToolContext, ToolError, ToolResult, tool
from baseagent.workspace import Workspace

JOB_TIMEOUT = 900
STOP_TIMEOUT = 30
SMOKE_TIMEOUT = 300  # the whole smoke run, on this machine's CPU
SMOKE_CELL_TIMEOUT = 180
SMOKE_DIR = "outputs/smoke"
FILENAME = re.compile(r"\w[\w.-]*")
GPU_LINE = re.compile(r"^BASEAGENT_GPU=(.+)$", re.MULTILINE)
END_MARKER = "BASEAGENT_NOTEBOOK_COMPLETE"
LOGGER = logging.getLogger(__name__)
# Kaggle credentials never belong in a notebook: public datasets download without them.
CREDENTIALS = re.compile(r"kaggle\.json|KAGGLE_(?:KEY|USERNAME|API_TOKEN)|\.kaggle/", re.IGNORECASE)
# Package installs are for Colab; the smoke run uses the packages installed here.
PIP_INSTALL = re.compile(r"^(\s*)[!%]\s*pip\s+install\b.*$", re.MULTILINE)

HARNESS = """\
import os
from pathlib import Path

# Set by BaseAgent. SMOKE is True only in its quick CPU check before the Colab run:
# then the notebook uses a tiny subset and one short epoch; everything else is the same.
SMOKE = os.environ.get("BASEAGENT_SMOKE") == "1"
# Write every file to return (weights, label maps, submissions) to OUTPUT_DIR.
OUTPUT_DIR = Path(os.environ.get("BASEAGENT_OUTPUT_DIR")
                  or ("/content" if Path("/content").is_dir() else "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("SMOKE:", SMOKE, "| OUTPUT_DIR:", OUTPUT_DIR)
"""

PREFLIGHT = """\
from pathlib import Path
from subprocess import check_output

gpu = check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=15,
)
Path("/content/data").mkdir(exist_ok=True)
print("BASEAGENT_GPU=" + gpu.splitlines()[0].strip())
"""


class NotebookCell(TypedDict):
    code: str
    note: NotRequired[str]


@dataclass
class ColabDraft:
    title: str
    path: Path
    kaggle_handle: str | None = None
    cells: list[NotebookCell] = field(default_factory=list)
    smoke_passed: str | None = None  # fingerprint of the draft that last passed a smoke run

    def attachment(self, workspace: Workspace) -> Attachment:
        return Attachment(kind="file", title=self.path.name, url=workspace.url(self.path), mime=NOTEBOOK_MIME)


def colab_workspace(ctx: ToolContext) -> Workspace:
    if ctx.agent.settings.notebook_runtime != "colab-t4":
        raise ToolError("Colab is off for this chat. Select Colab T4 under Notebook compute in Settings.")
    if ctx.agent.workspace is None:
        raise ToolError("This chat has no data workspace.")
    return ctx.agent.workspace


@tool(exclusive=True)
def start_colab_notebook(title: str, kaggle_source: str | None = None, ctx: ToolContext = None) -> str:
    """Start a new draft for a Colab notebook. It will be filled one cell at a time.

    Args:
        title: A short notebook title, such as Full-card CNN baseline.
        kaggle_source: The public Kaggle dataset URL or owner/dataset handle, if the user supplied one.
            The notebook gets a download cell and a dataset_root variable automatically.
    """
    workspace = colab_workspace(ctx)
    handle = kaggle_dataset(kaggle_source) if kaggle_source else ctx.state.get("requested_kaggle_dataset")
    folder = f"outputs/colab-draft-{uuid4().hex[:12]}"
    ctx.state["colab_draft"] = ColabDraft(title.strip() or "Data science notebook",
                                         workspace.resolve(f"{folder}/colab_draft.ipynb"), handle)
    ctx.state.pop("colab_outputs", None)
    message = ("Started a Colab draft. Use save_colab_cell for the first authored cell. Cells written by "
               "BaseAgent come first and define SMOKE (bool) and OUTPUT_DIR (pathlib.Path).")
    if handle:
        message += (f" They also download {handle} and define dataset_root (pathlib.Path). If you "
                    "haven't yet, call inspect_dataset to see what it contains.")
    return message


HARNESS_NAMES = ("SMOKE", "OUTPUT_DIR")  # plus dataset_root when the draft downloads a Kaggle dataset


def rebound_harness_names(tree: ast.AST, names: tuple[str, ...]) -> list[tuple[str, int]]:
    """Places a cell assigns, deletes or declares global one of BaseAgent's variables."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in names:
            found.append((node.id, node.lineno))
        elif isinstance(node, ast.Global):
            found += [(name, node.lineno) for name in node.names if name in names]
    return sorted(set(found), key=lambda item: item[1])


def validate_colab_cell(code: str, number: int, harness_names: tuple[str, ...] = HARNESS_NAMES) -> None:
    if CREDENTIALS.search(code):
        raise ToolError("Don't use Kaggle credentials in the notebook. The download cell gets public "
                        "datasets without them, as dataset_root; if Kaggle denies access, tell the user.")
    try:
        tree = ast.parse(TransformerManager().transform_cell(code))
    except SyntaxError as exc:
        raise ToolError(f"Cell {number} has invalid Python at line {exc.lineno}: {exc.msg}.") from exc
    rebound = rebound_harness_names(tree, harness_names)
    if rebound:
        where = ", ".join(f"{name} (line {line})" for name, line in rebound)
        raise ToolError(f"Cell {number} reassigns {where}. BaseAgent's cells define "
                        f"{', '.join(harness_names)} in every run, smoke and Colab alike: use them as they "
                        "are, with no fallback such as try/except NameError, and pick other names for "
                        "your own variables.")
    for node in ast.walk(tree):
        invalid_import = (isinstance(node, ast.ImportFrom) and node.module == "kagglehub"
                          and any(name.name == "KaggleHub" for name in node.names))
        invalid_attribute = (isinstance(node, ast.Attribute) and node.attr == "KaggleHub"
                             and isinstance(node.value, ast.Name) and node.value.id == "kagglehub")
        if invalid_import or invalid_attribute:
            raise ToolError("kagglehub has no KaggleHub class. Pass the dataset URL to start_colab_notebook "
                            "and use dataset_root in your cells.")


@tool(exclusive=True)
def save_colab_cell(number: int, code: str, note: str = "", ctx: ToolContext = None) -> ToolResult:
    """Save one Python cell in the current Colab draft without executing it.

    Args:
        number: One-based position among the authored cells. Use the next number to append,
            or an existing number to replace a cell.
        code: Python source for this cell only.
        note: Optional Markdown heading and explanation for the cell.
    """
    workspace = colab_workspace(ctx)
    draft = ctx.state.get("colab_draft")
    if draft is None:
        raise ToolError("Call start_colab_notebook before saving a cell.")
    if not code.strip():
        raise ToolError("Provide code for this cell.")
    if not 1 <= number <= len(draft.cells) + 1:
        raise ToolError(f"Choose a cell number from 1 to {len(draft.cells) + 1}.")

    validate_colab_cell(code, number, HARNESS_NAMES + (("dataset_root",) if draft.kaggle_handle else ()))
    cells = list(draft.cells)
    cells[number - 1:number] = [{"code": code, "note": note}]
    notebook = make_notebook(cells, [], title=draft.title, kaggle_handle=draft.kaggle_handle)
    draft.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = draft.path.with_suffix(".tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(draft.path)
    draft.cells = cells

    attachments = [draft.attachment(workspace)]
    ctx.state["colab_outputs"] = attachments
    return ToolResult(f"Saved cell {number}. The draft contains {len(cells)} authored cells and has not run.",
                      attachments=attachments)


def input_files(workspace: Workspace, inputs: list[str]) -> dict[str, Path]:
    files = {}
    for relative in inputs:
        prefix, _, name = relative.partition("/")
        if prefix != "data" or not FILENAME.fullmatch(name):
            raise ToolError(f"Colab inputs must be uploaded files under data/: {relative!r}.")
        path = workspace.resolve(relative)
        if path != workspace.root / relative or not path.is_file():
            raise ToolError(f"Upload {relative!r} to this chat before running Colab.")
        files[relative] = path
    return files


async def cli(binary: str, directory: Path, name: str, action: str, *args: str) -> str:
    """Run the CLI locally with bounded diagnostics and no interactive stdin."""
    env = {key: value for key, value in os.environ.items() if key in {
        "HOME", "PATH", "CLOUDSDK_CONFIG", "GOOGLE_APPLICATION_CREDENTIALS",
        "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    }}
    try:
        process = await asyncio.create_subprocess_exec(
            binary, "--auth", "adc", action, "-s", name, *args,
            cwd=directory, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        raise ToolError("Could not start the Colab CLI. Install it with make colab-setup.") from exc

    tail = b""
    try:
        assert process.stdout is not None
        while chunk := await process.stdout.read(8192):
            tail = (tail + chunk)[-8192:]
        status = await process.wait()
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.communicate()
        raise

    output = tail.decode("utf-8", errors="replace")
    if status:
        diagnostic = output.lower()
        if any(word in diagnostic for word in ("credential", "unauthorized", "permission_denied", "insufficient scope")):
            raise ToolError("Colab authentication failed. Configure Google Application Default Credentials "
                            "with the Colab scope, then run colab --auth adc usage.")
        if action == "new":
            raise ToolError("Colab could not provision a T4. Check your account's runtime availability and quota.")
        if action in {"upload", "download"}:
            filename = Path(args[-1] if action == "upload" else args[0]).name
            raise ToolError(f"Colab {action} failed for {filename!r}. Check the file path and size.")
        raise ToolError(f"Colab {action} failed (exit {status}).")
    return output


async def release_runtime(binary: str, directory: Path, name: str) -> str | None:
    try:
        await asyncio.wait_for(cli(binary, directory, name, "stop"), STOP_TIMEOUT)
    except (ToolError, TimeoutError):
        warning = f"Colab cleanup could not be confirmed. Check colab --auth adc stop -s {name}."
        LOGGER.warning("%s", warning)
        return warning
    return None


def make_notebook(cells: list[NotebookCell], inputs: list[str], hardware: str | None = None,
                  title: str = "Data science notebook", kaggle_handle: str | None = None) -> nbformat.NotebookNode:
    recorder = NotebookRecorder()
    status = f"Executed on Google Colab using {hardware}." if hardware else "Colab draft. The cells have not run."
    recorder.add_markdown(f"# {title}\n\n{status}")
    recorder.add_code(SETUP_CODE, [], None)
    recorder.add_markdown("## Notebook settings")
    recorder.add_code(HARNESS, [], None)
    if kaggle_handle:
        recorder.add_markdown("## Download the Kaggle dataset")
        recorder.add_code(kaggle_download_source(kaggle_handle), [], None)
    for number, cell in enumerate(cells, start=1):
        recorder.add_markdown(cell.get("note", ""))
        recorder.add_code(cell["code"], [], None)
        recorder.cells[-1].metadata["baseagent_cell"] = number
    if hardware:
        recorder.add_code(f"print({END_MARKER!r})", [], None)
    return recorder.export(inputs)


def fingerprint(draft: ColabDraft, inputs: list[str], outputs: list[str]) -> str:
    """Identifies exactly what a smoke run checked: the cells, the dataset and the files."""
    payload = json.dumps([draft.kaggle_handle, draft.cells, sorted(set(inputs)), sorted(set(outputs))])
    return hashlib.sha256(payload.encode()).hexdigest()


def check_outputs(outputs: list[str]) -> None:
    if any(not FILENAME.fullmatch(name) or name == "analysis.ipynb" for name in outputs):
        raise ToolError("Use plain output filenames such as submission.csv. analysis.ipynb is reserved.")


def smoke_source(code: str) -> str:
    return PIP_INSTALL.sub(lambda m: f"{m[1]}pass  # pip install skipped in the smoke run", code)


def current_draft(ctx: ToolContext) -> ColabDraft:
    draft = ctx.state.get("colab_draft")
    if draft is None or not draft.cells:
        raise ToolError("Start a Colab notebook draft and save its cells before running it.")
    return draft


@tool(exclusive=True, timeout=SMOKE_TIMEOUT + DOWNLOAD_TIMEOUT + 60)
async def smoke_test_notebook(inputs: list[str], outputs: list[str], ctx: ToolContext = None) -> ToolResult:
    """Run the whole draft here on CPU with SMOKE = True: the check before Colab.

    Every cell runs from the top in a fresh kernel, as on Colab, with SMOKE True, so
    the notebook must then use a tiny subset of the data and one short epoch. Fix any
    failing cell and run this again. run_colab_notebook only accepts a draft that
    passed, unchanged, with the same inputs and outputs.

    Args:
        inputs: Uploaded chat paths the notebook reads, such as data/train.csv. Use [] for none.
        outputs: Filenames the notebook writes to OUTPUT_DIR, such as model.pt. Each must exist afterwards.
    """
    workspace = colab_workspace(ctx)
    draft = current_draft(ctx)
    draft.smoke_passed = None
    check_outputs(outputs)
    files = input_files(workspace, inputs)

    output_dir = workspace.resolve(SMOKE_DIR)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    env = {"BASEAGENT_SMOKE": "1", "BASEAGENT_OUTPUT_DIR": str(output_dir), "MPLBACKEND": "Agg"}
    if draft.kaggle_handle:
        env["BASEAGENT_DATASET_ROOT"] = str(await ensure_dataset(draft.kaggle_handle, ctx))

    notebook = make_notebook(draft.cells, list(files), title=draft.title, kaggle_handle=draft.kaggle_handle)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    started = time.monotonic()
    seen: list[str] = []
    kernel = KernelSandbox()
    await kernel.start(workspace.root, env)
    try:
        for index, cell in enumerate(code_cells, start=1):
            number = cell.metadata.get("baseagent_cell")
            where = f"authored cell {number}" if number else f"setup cell {index} (written by BaseAgent)"
            remaining = SMOKE_TIMEOUT - (time.monotonic() - started)
            if remaining < 5:
                raise ToolError(f"The smoke run used its {SMOKE_TIMEOUT} s before {where}. When SMOKE is True, "
                                "use a much smaller subset and fewer batches.")
            await ctx.progress(f"Smoke run: {where}")
            result = await kernel.execute(smoke_source(cell.source), timeout=min(SMOKE_CELL_TIMEOUT, remaining))
            text = result.text()
            if not result.ok:
                message = f"Smoke run failed at {where}: {result.error or result.status}\n{text[-1500:]}"
                if result.status in {"timeout", "restarted"}:
                    message += ("\nWhen SMOKE is True, use a tiny subset (a few examples per class) and "
                                "one short epoch.")
                if seen:
                    message += "\n\nOutput of the cells before it:\n" + "\n".join(seen[-3:])[-1500:]
                raise ToolError(message)
            if text:
                seen.append(f"[{where}] {text[:500]}")
    finally:
        await kernel.shutdown()

    missing = [name for name in dict.fromkeys(outputs) if not (output_dir / name).is_file()]
    if missing:
        raise ToolError(f"The smoke run finished, but OUTPUT_DIR has no {', '.join(missing)}. Write each "
                        "declared output to OUTPUT_DIR / filename, also when SMOKE is True.")
    draft.smoke_passed = fingerprint(draft, inputs, outputs)
    elapsed = time.monotonic() - started
    found = f" Outputs written: {', '.join(dict.fromkeys(outputs))}." if outputs else ""
    summary = "\n".join(seen)[-3500:]
    return ToolResult(f"Smoke run passed in {elapsed:.0f} s: all {len(code_cells)} code cells ran on CPU "
                      f"with SMOKE = True.{found} Next: run_colab_notebook with the same inputs and outputs. "
                      f"Smoke numbers come from a tiny subset; don't report them as results.\n\n{summary}",
                      attachments=[draft.attachment(workspace)])


def read_execution(path: Path, source: nbformat.NotebookNode) -> tuple[nbformat.NotebookNode, str]:
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except (OSError, ValueError, nbformat.reader.NotJSONError, nbformat.ValidationError) as exc:
        raise ToolError("Colab did not return a valid executed notebook.") from exc

    if [(c.cell_type, c.source) for c in notebook.cells] != [(c.cell_type, c.source) for c in source.cells]:
        raise ToolError("The notebook returned by Colab does not match the submitted cells.")
    marker = notebook.cells.pop()
    observations = []
    for index, cell in enumerate((c for c in notebook.cells if c.cell_type == "code"), start=1):
        result = CellResult(index, cell.outputs)
        error = next((out for out in cell.outputs if out.output_type == "error"), None)
        if error is not None:
            number = cell.metadata.get("baseagent_cell")
            location = f" (authored cell {number})" if number is not None else ""
            raise ToolError(f"Colab cell {index}{location} failed: {error.ename}: {error.evalue[:400]}\n"
                            f"{result.text()[-1400:]}")
        cell.execution_count = index
        if result.text():
            observations.append(f"Cell {index}: {result.text()[:800]}")
    if CellResult(None, marker.outputs).text() != END_MARKER:
        raise ToolError("Colab did not complete the notebook.")
    return notebook, "\n\n".join(observations)[-5000:]


@tool(exclusive=True, timeout=JOB_TIMEOUT + STOP_TIMEOUT + 10)
async def run_colab_notebook(inputs: list[str], outputs: list[str], ctx: ToolContext = None) -> ToolResult:
    """Execute the saved draft on a verified Colab T4 and return its files.

    Only a draft that passed smoke_test_notebook, unchanged, with the same inputs and outputs.

    Args:
        inputs: Uploaded chat paths the notebook reads, such as data/train.csv. Use [] for none.
        outputs: Filenames the notebook writes to OUTPUT_DIR (/content on Colab), such as submission.csv.
    """
    workspace = colab_workspace(ctx)
    draft = current_draft(ctx)
    ctx.state["colab_outputs"] = [draft.attachment(workspace)]
    check_outputs(outputs)
    if draft.smoke_passed != fingerprint(draft, inputs, outputs):
        raise ToolError("Run smoke_test_notebook first, with these inputs and outputs. Colab only runs a "
                        "draft that passed a smoke run after its last change.")

    binary = shutil.which("colab")
    if not binary:
        raise ToolError("Install the Colab CLI on the agent server with make colab-setup.")
    try:
        files = input_files(workspace, inputs)
    except ValueError as exc:
        raise ToolError("Colab inputs must be uploaded files inside this chat's data folder.") from exc

    name = f"baseagent-{uuid4().hex[:12]}"
    with TemporaryDirectory(dir=workspace.ensure(), prefix=".colab-") as temporary:
        directory = Path(temporary)
        cleanup_warning = None
        try:
            async with asyncio.timeout(JOB_TIMEOUT):
                await ctx.progress("Starting a Colab T4 runtime")
                await cli(binary, directory, name, "new", "--gpu", "T4")

                preflight = directory / "verify.py"
                preflight.write_text(PREFLIGHT, encoding="utf-8")
                result = await cli(binary, directory, name, "exec", "-f", preflight.name, "--timeout", "60")
                gpu = GPU_LINE.search(result)
                if not gpu or not re.search(r"\bT4\b", gpu[1]):
                    raise ToolError("The Colab runtime did not report a T4 GPU.")
                hardware = gpu[1].strip()

                for relative, path in files.items():
                    await ctx.progress(f"Uploading {path.name} to Colab")
                    await cli(binary, directory, name, "upload", str(path), f"/content/{relative}")

                source = make_notebook(draft.cells, list(files), hardware, draft.title, draft.kaggle_handle)
                nbformat.write(source, directory / "analysis.ipynb")
                await ctx.progress(f"Running the notebook on {hardware}")
                await cli(binary, directory, name, "exec", "-f", "analysis.ipynb", "--timeout", "0")
                notebook, summary = read_execution(directory / "analysis_output.ipynb", source)

                await ctx.progress("Retrieving the Colab notebook and outputs")
                ready = directory / "results"
                ready.mkdir()
                nbformat.write(notebook, ready / "analysis.ipynb")
                for filename in dict.fromkeys(outputs):
                    await cli(binary, directory, name, "download", f"/content/{filename}", str(ready / filename))
        except TimeoutError as exc:
            raise ToolError(f"The Colab run exceeded the {JOB_TIMEOUT // 60}-minute limit.") from exc
        finally:
            cleanup = asyncio.create_task(release_runtime(binary, directory, name))
            try:
                cleanup_warning = await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            if cleanup_warning:
                await ctx.progress(cleanup_warning)

        target = workspace.resolve(f"outputs/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish the downloads together after every requested file has arrived.
        ready.rename(target)

    attachments = [
        Attachment(kind="file", title=path.name, url=workspace.url(path),
                   mime=NOTEBOOK_MIME if path.suffix == ".ipynb" else
                   mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        for path in (target / filename for filename in ("analysis.ipynb", *dict.fromkeys(outputs)))
    ]
    ctx.state["colab_outputs"] = attachments
    message = f"Notebook completed on Colab ({hardware})."
    if summary:
        message += "\n\n" + summary
    if cleanup_warning:
        message += "\n\n" + cleanup_warning
    return ToolResult(message, attachments=attachments)
