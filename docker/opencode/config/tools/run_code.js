// run_code: the fixer's only way to run code.
//
// It reads a file from the workspace and asks BaseAgent's runner bridge to
// check or run it in a separate, throwaway container (no network, 10 s,
// 512 MB). The model has no shell, so this is the only execution path, and
// BaseAgent logs every run itself.
//
// Configured by two environment variables BaseAgent gives the OpenCode server:
//   BASEAGENT_RUN_URL    e.g. http://host.docker.internal:8000/internal/code-runs
//   BASEAGENT_RUN_TOKEN  a one-time token for this request (a few runs only)
//
// Plain object with JSON Schema args, so no npm packages are needed.

import { readFile, realpath } from "node:fs/promises"
import path from "node:path"

const LANGUAGES = { ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript" }
const MAX_BYTES = 200_000
const OUTPUT_CHARS = 3000

function clip(text) {
  if (!text) return ""
  return text.length <= OUTPUT_CHARS ? text : text.slice(0, OUTPUT_CHARS) + `\n... [${text.length - OUTPUT_CHARS} more characters]`
}

function describe(result) {
  const lines = [`status: ${result.status}`, `note: ${result.note}`]
  if (result.exit_code !== null && result.exit_code !== undefined) lines.push(`exit code: ${result.exit_code}`)
  lines.push(`duration: ${result.duration} s`)
  if (result.stdout) lines.push("stdout:", clip(result.stdout))
  if (result.stderr) lines.push("stderr:", clip(result.stderr))
  return lines.join("\n")
}

export default {
  description:
    "Check or run a Python, JavaScript or TypeScript file from the workspace in a separate sandbox " +
    "(no network, no installs, 10 seconds, 512 MB). mode 'check' only checks syntax and obvious errors " +
    "(Python: compile + undefined names; JavaScript: syntax; TypeScript: type check). mode 'run' executes " +
    "the file. Run limits are counted, so check first, then run once the code should work.",
  args: {
    path: { type: "string", description: "File to check or run, relative to the workspace, e.g. main.py" },
    mode: { type: "string", enum: ["check", "run"], description: "check: syntax and type check only. run: execute it." },
    stdin: { type: "string", description: "Text for the program's standard input. Use \"\" when it reads no input." },
  },
  async execute(args, context) {
    const url = process.env.BASEAGENT_RUN_URL
    const token = process.env.BASEAGENT_RUN_TOKEN
    if (!url || !token) return "run_code is not available: the sandbox has no runner configured."

    const mode = args.mode === "check" ? "check" : args.mode === "run" ? "run" : null
    if (!mode) return "mode must be 'check' or 'run'."
    if (typeof args.path !== "string" || !args.path.trim()) return "path is required, e.g. main.py"
    const stdin = typeof args.stdin === "string" ? args.stdin : ""

    const root = await realpath(context.directory)
    let file
    try {
      file = await realpath(path.resolve(root, args.path))
    } catch {
      return `No such file: ${args.path}`
    }
    if (file !== root && !file.startsWith(root + path.sep)) return "Only files inside the workspace can be run."
    const language = LANGUAGES[path.extname(file).toLowerCase()]
    if (!language) return `Can't run ${path.basename(file)}: only .py, .js, .mjs, .cjs and .ts files can be run.`

    const code = await readFile(file, "utf8")
    if (Buffer.byteLength(code) > MAX_BYTES) return `${args.path} is too large to run (over ${MAX_BYTES} bytes).`

    let response
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ language, code, mode, stdin }),
        signal: context.abort,
      })
    } catch (error) {
      return `run_code could not reach the runner: ${error?.message ?? error}`
    }
    const text = await response.text()
    if (!response.ok) {
      let detail = text
      try {
        detail = JSON.parse(text).detail ?? text
      } catch {}
      return `run_code was refused (HTTP ${response.status}): ${detail}`
    }
    const result = JSON.parse(text)
    return {
      title: `${mode} ${path.relative(root, file)}: ${result.status}`,
      output: describe(result),
      metadata: { status: result.status, mode, language, duration: result.duration },
    }
  },
}
