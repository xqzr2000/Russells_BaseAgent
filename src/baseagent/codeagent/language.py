"""Guess the language of a code snippet that came without a ``` tag or file name.

Scores simple, telling markers per language and returns the best match, or
None when nothing stands out. The chat composer uses the same idea
(web/src/codeLanguage.ts) to tag pasted code, so this is the fallback for
code that still arrives unlabelled.
"""

from __future__ import annotations

import re

# (pattern, weight) per language. Multiline patterns use ^/$ per line.
_MARKERS: dict[str, list[tuple[str, int]]] = {
    "typescript": [
        (r":\s*(?:string|number|boolean|any|void|unknown|never)(?:\[\])?\s*[=,;)\]{]", 4),
        (r"^\s*(?:export\s+)?interface\s+\w+", 4), (r"^\s*(?:export\s+)?type\s+\w+\s*=", 4),
        (r"\bas\s+(?:const|string|number|any)\b", 3), (r"\bimport\s+type\b", 4),
        (r"\b(?:public|private|readonly)\s+\w+\s*:", 3),
    ],
    "javascript": [
        (r"^\s*(?:const|let|var)\s+[\w{\[]", 3), (r"\bfunction\b\s*\w*\s*\(", 2), (r"=>", 2),
        (r"\bdocument\.", 3), (r"\bconsole\.log\(", 3), (r"\bwindow\.", 2), (r"===|!==", 2),
        (r"\brequire\(|\bmodule\.exports\b", 3), (r"\baddEventListener\(", 3),
        (r";\s*$", 1), (r"^\s*//", 1), (r"\bnew Promise\b|\bawait\s+fetch\(", 2),
    ],
    "python": [
        (r"^\s*def\s+\w+\s*\(", 3), (r"^\s*def\s+\w+\s*\(.*\)\s*(?:->\s*[\w\[\], .]+)?:\s*$", 2), (r"^\s*(?:import\s+\w+|from\s+\w[\w.]*\s+import\b)", 3),
        (r"\bprint\(", 2), (r"^\s*elif\b", 4), (r"^\s*(?:if|for|while|with|try|else|class)\b.*:\s*$", 2),
        (r"\bself\.", 2), (r"\b(?:None|True|False)\b", 1), (r"^\s*#(?!include|!)", 1),
        (r"\b(?:range|len)\(", 1), (r"f\"[^\"]*\{", 2),
    ],
    "java": [(r"\bpublic\s+(?:static\s+)?(?:class|void)\b", 4), (r"\bSystem\.out\.print", 5), (r"\bString\[\]\s+args\b", 5)],
    "c": [(r"^\s*#include\s*<\w+\.h>", 5), (r"\bprintf\(", 2), (r"\bint\s+main\s*\(", 3)],
    "cpp": [(r"^\s*#include\s*<(?:iostream|vector|string)>", 6), (r"\bstd::", 4), (r"\bcout\s*<<", 4)],
    "go": [(r"^\s*package\s+\w+", 4), (r"^\s*func\s+\w+\s*\(", 3), (r"\bfmt\.", 4), (r":=", 2)],
    "rust": [(r"\bfn\s+\w+\s*\(", 3), (r"\blet\s+mut\b", 4), (r"\bprintln!\(", 5), (r"->\s*\w+\s*\{", 1)],
    "ruby": [(r"^\s*puts\b", 3), (r"^\s*end\s*$", 2), (r"^\s*def\s+\w+[^:]*$", 2), (r"\.each\s+do\b", 4)],
    "php": [(r"<\?php", 8), (r"\$\w+\s*=", 2), (r"\becho\b", 1)],
    "sql": [(r"(?i)^\s*select\b.+\bfrom\b", 4), (r"(?i)^\s*(?:insert\s+into|create\s+table|update\s+\w+\s+set)\b", 5)],
    "bash": [(r"^#!.*\b(?:bash|sh)\b", 6), (r"^\s*echo\s+[\"$]", 2), (r"\$\{?\w+\}?", 1), (r"^\s*fi\s*$", 3)],
}
_COMPILED = {lang: [(re.compile(p, re.MULTILINE), w) for p, w in markers] for lang, markers in _MARKERS.items()}
MIN_SCORE = 2


def language_scores(code: str) -> dict[str, int]:
    scores = {lang: sum(w for rx, w in markers if rx.search(code)) for lang, markers in _COMPILED.items()}
    # TypeScript is JavaScript plus types: only count it when it has its own markers.
    if scores["typescript"]:
        scores["typescript"] += scores["javascript"]
    return scores


def guess_language(code: str) -> str | None:
    """"python", "javascript", "typescript", ... or None if unclear."""
    scores = language_scores(code)
    best = max(scores, key=lambda lang: scores[lang])
    return best if scores[best] >= MIN_SCORE else None
