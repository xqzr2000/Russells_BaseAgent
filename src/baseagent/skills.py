"""Skills with progressive disclosure.

A skill is a folder holding a ``SKILL.md`` whose YAML frontmatter has a
``name`` and ``description``. Only those two lines go into the system prompt
(the catalog); the full body is returned when the model calls ``invoke_skill``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from baseagent.tools import Tool, ToolError, tool


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path

    @property
    def metadata(self) -> str:
        return f"name: {self.name}\ndescription: {self.description}"


def parse_frontmatter(text: str, source: Path) -> dict[str, str]:
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source}: missing YAML frontmatter (expected a leading '---')")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError(f"{source}: frontmatter is not closed by '---'") from None
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: malformed YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{source}: frontmatter must be a mapping")
    parsed = {}
    for key in ("name", "description"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: frontmatter needs a non-empty string {key!r}")
        parsed[key] = " ".join(value.split())
    return parsed


class SkillLibrary:
    def __init__(self, skills: dict[str, Skill] | None = None):
        self.skills = skills or {}

    @classmethod
    def load(cls, *paths: str | Path | None) -> "SkillLibrary":
        skills: dict[str, Skill] = {}
        for root in paths:
            if root is None:
                continue
            root = Path(root)
            if not root.is_dir():
                raise ValueError(f"Skills path is not a directory: {root}")
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                if folder.name.startswith((".", "_")):
                    continue
                file = folder / "SKILL.md"
                if not file.is_file():
                    raise ValueError(f"Skill folder {folder} has no SKILL.md")
                text = file.read_text(encoding="utf-8")
                meta = parse_frontmatter(text, file)
                if meta["name"] in skills:
                    raise ValueError(f"Duplicate skill name {meta['name']!r} in {file}")
                skills[meta["name"]] = Skill(meta["name"], meta["description"], text, file)
        return cls(skills)

    def __bool__(self) -> bool:
        return bool(self.skills)

    def catalog_prompt(self) -> str:
        catalog = "\n\n".join(skill.metadata for skill in self.skills.values())
        return (
            "# Skills\n"
            "Skills are reusable instructions for specific kinds of work. Only their "
            "names and descriptions are listed here. Before doing work a skill covers, "
            "call `invoke_skill` with its name to load the full instructions, then "
            f"follow them.\n\n<skills>\n{catalog}\n</skills>"
        )

    def as_tool(self) -> Tool:
        library = self

        @tool(name="invoke_skill")
        def invoke_skill(name: str) -> str:
            """Load a skill's full instructions. Call this before work the skill covers.

            Args:
                name: The skill's name from the skills catalog.
            """
            skill = library.skills.get(name.strip())
            if skill is None:
                raise ToolError(
                    f"Unknown skill {name!r}. Available: {', '.join(library.skills) or '(none)'}."
                )
            return skill.content

        return invoke_skill
