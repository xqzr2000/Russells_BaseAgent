// Recognise pasted code and wrap it in a ``` block with its language, so the code agent
// reads it straight from the message (the coordinator never has to copy it). The same
// markers as src/baseagent/codeagent/language.py, which handles code that still arrives
// unlabelled.

type Marker = [RegExp, number];

const MARKERS: Record<string, Marker[]> = {
  typescript: [
    [/:\s*(?:string|number|boolean|any|void|unknown|never)(?:\[\])?\s*[=,;)\]{]/m, 4],
    [/^\s*(?:export\s+)?interface\s+\w+/m, 4],
    [/^\s*(?:export\s+)?type\s+\w+\s*=/m, 4],
    [/\bas\s+(?:const|string|number|any)\b/m, 3],
    [/\bimport\s+type\b/m, 4],
    [/\b(?:public|private|readonly)\s+\w+\s*:/m, 3],
  ],
  javascript: [
    [/^\s*(?:const|let|var)\s+[\w{[]/m, 3],
    [/\bfunction\b\s*\w*\s*\(/m, 2],
    [/=>/m, 2],
    [/\bdocument\./m, 3],
    [/\bconsole\.log\(/m, 3],
    [/\bwindow\./m, 2],
    [/===|!==/m, 2],
    [/\brequire\(|\bmodule\.exports\b/m, 3],
    [/\baddEventListener\(/m, 3],
    [/;\s*$/m, 1],
    [/^\s*\/\//m, 1],
    [/\bnew Promise\b|\bawait\s+fetch\(/m, 2],
  ],
  python: [
    [/^\s*def\s+\w+\s*\(/m, 3],
    [/^\s*def\s+\w+\s*\(.*\)\s*(?:->\s*[\w[\], .]+)?:\s*$/m, 2],
    [/^\s*(?:import\s+\w+|from\s+\w[\w.]*\s+import\b)/m, 3],
    [/\bprint\(/m, 2],
    [/^\s*elif\b/m, 4],
    [/^\s*(?:if|for|while|with|try|else|class)\b.*:\s*$/m, 2],
    [/\bself\./m, 2],
    [/\b(?:None|True|False)\b/m, 1],
    [/^\s*#(?!include|!)/m, 1],
    [/\b(?:range|len)\(/m, 1],
    [/f"[^"]*\{/m, 2],
  ],
  java: [[/\bpublic\s+(?:static\s+)?(?:class|void)\b/m, 4], [/\bSystem\.out\.print/m, 5], [/\bString\[\]\s+args\b/m, 5]],
  c: [[/^\s*#include\s*<\w+\.h>/m, 5], [/\bprintf\(/m, 2], [/\bint\s+main\s*\(/m, 3]],
  cpp: [[/^\s*#include\s*<(?:iostream|vector|string)>/m, 6], [/\bstd::/m, 4], [/\bcout\s*<</m, 4]],
  go: [[/^\s*package\s+\w+/m, 4], [/^\s*func\s+\w+\s*\(/m, 3], [/\bfmt\./m, 4], [/:=/m, 2]],
  rust: [[/\bfn\s+\w+\s*\(/m, 3], [/\blet\s+mut\b/m, 4], [/\bprintln!\(/m, 5], [/->\s*\w+\s*\{/m, 1]],
  ruby: [[/^\s*puts\b/m, 3], [/^\s*end\s*$/m, 2], [/^\s*def\s+\w+[^:]*$/m, 2], [/\.each\s+do\b/m, 4]],
  php: [[/<\?php/m, 8], [/\$\w+\s*=/m, 2], [/\becho\b/m, 1]],
  sql: [[/^\s*select\b.+\bfrom\b/im, 4], [/^\s*(?:insert\s+into|create\s+table|update\s+\w+\s+set)\b/im, 5]],
  bash: [[/^#!.*\b(?:bash|sh)\b/m, 6], [/^\s*echo\s+["$]/m, 2], [/\$\{?\w+\}?/m, 1], [/^\s*fi\s*$/m, 3]],
};

// Pasting is stricter than the server's guess (2): prose must never be wrapped.
const PASTE_MIN_SCORE = 3;

export function guessLanguage(code: string, minScore = 2): string | null {
  const scores: Record<string, number> = {};
  for (const [lang, markers] of Object.entries(MARKERS)) {
    scores[lang] = markers.reduce((sum, [rx, weight]) => sum + (rx.test(code) ? weight : 0), 0);
  }
  if (scores.typescript) scores.typescript += scores.javascript;
  let best: string | null = null;
  for (const lang of Object.keys(scores)) if (best === null || scores[lang] > scores[best]) best = lang;
  return best !== null && scores[best] >= minScore ? best : null;
}

/** The pasted text wrapped in a ``` block if it is multi-line code; otherwise null. */
export function fenceIfCode(text: string): string | null {
  const trimmed = text.replace(/^\n+|\s+$/g, "");
  if (trimmed.includes("```") || trimmed.split("\n").length < 2) return null;
  const lang = guessLanguage(trimmed, PASTE_MIN_SCORE);
  return lang ? "```" + lang + "\n" + trimmed + "\n```" : null;
}
