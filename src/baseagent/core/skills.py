"""Skills with progressive disclosure.

A skill is a folder holding a ``SKILL.md`` whose YAML frontmatter has a
``name`` and ``description``. Only a short entry per skill goes into the system
prompt (the catalog); the full body is returned when the model calls
``invoke_skill``.

A skills root holds skill folders, area folders, or both. An area folder has a
``GROUP.md`` (frontmatter ``title`` and ``description``) and skill folders
inside it; the catalog lists skills under their area. A skill whose body has a
``## Scenarios it explains`` section gets a ``scenarios:`` line in the catalog
(its bold quoted symptoms), so a question that describes a symptom without
naming the method still finds the skill.

Roots are required (an error stops the agent) or optional (missing roots and
broken skills are skipped with a warning): the private knowledge submodule is
optional, so BaseAgent works without it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from baseagent.core.tools import Tool, ToolError, tool

LOGGER = logging.getLogger(__name__)
SCENARIO = re.compile(r'^\s*[-*]\s+\*\*"?(.+?)"?\*\*', re.MULTILINE)
SCENARIO_CHARS = 240  # per skill: the catalog is in every prompt
MAX_SKILL_FILE = 60_000


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path
    area: str | None = None  # the area's title, for skills inside an area folder
    scenarios: str = ""

    @property
    def metadata(self) -> str:
        lines = [f"name: {self.name}", f"description: {self.description}"]
        if self.scenarios:
            lines.append(f"scenarios: {self.scenarios}")
        return "\n".join(lines)


def parse_frontmatter(text: str, source: Path, keys: tuple[str, ...] = ("name", "description")) -> dict[str, str]:
    lines = text.lstrip("\ufeff").splitlines()
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
    for key in keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: frontmatter needs a non-empty string {key!r}")
        parsed[key] = " ".join(value.split())
    return parsed


def scenarios_of(text: str) -> str:
    """The bold symptoms under '## Scenarios it explains', joined into one short line."""
    match = re.search(r"^## +Scenarios it explains\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    items = [" ".join(item.split()).rstrip(".") for item in SCENARIO.findall(match.group(1))]
    line = "; ".join(f'"{item}"' for item in items)
    return line if len(line) <= SCENARIO_CHARS else line[:SCENARIO_CHARS].rsplit(";", 1)[0] + "; …"


def read_skill(file: Path, area: str | None) -> Skill:
    text = file.read_text(encoding="utf-8")
    meta = parse_frontmatter(text, file)
    return Skill(meta["name"], meta["description"], text, file, area, scenarios_of(text))


def skills_in(root: Path) -> Iterable[tuple[Path, str | None, str | None]]:
    """(SKILL.md, area title, area description) for every skill under root, in order."""
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name.startswith((".", "_")):
            continue
        if (folder / "SKILL.md").is_file():
            yield folder / "SKILL.md", None, None
            continue
        group = folder / "GROUP.md"
        if not group.is_file():
            raise ValueError(f"Skill folder {folder} has no SKILL.md")
        meta = parse_frontmatter(group.read_text(encoding="utf-8"), group, ("title", "description"))
        for skill in sorted(p for p in folder.iterdir() if p.is_dir()):
            if skill.name.startswith((".", "_")):
                continue
            if not (skill / "SKILL.md").is_file():
                raise ValueError(f"Skill folder {skill} has no SKILL.md")
            yield skill / "SKILL.md", meta["title"], meta["description"]


class SkillLibrary:
    def __init__(self, skills: dict[str, Skill] | None = None, areas: dict[str, str] | None = None):
        self.skills = skills or {}
        self.areas = areas or {}  # area title -> description, in load order

    @classmethod
    def load(cls, *paths: str | Path | None, optional: Iterable[str | Path] = ()) -> "SkillLibrary":
        """Load every skill under the given roots.

        ``paths`` are required: a missing root or a broken skill raises ValueError.
        ``optional`` roots are skipped when missing, and their broken skills are skipped
        with a warning, so one bad file can't stop the agent.
        """
        skills: dict[str, Skill] = {}
        areas: dict[str, str] = {}
        roots = [(Path(p), False) for p in paths if p is not None] + [(Path(p), True) for p in optional]
        for root, is_optional in roots:
            if not root.is_dir():
                if is_optional:
                    LOGGER.info("Optional skills folder %s is not available; skipping it", root)
                    continue
                raise ValueError(f"Skills path is not a directory: {root}")
            try:
                entries = list(skills_in(root))
            except ValueError as exc:
                if not is_optional:
                    raise
                LOGGER.warning("Skipping optional skills folder %s: %s", root, exc)
                continue
            for file, area, description in entries:
                try:
                    skill = read_skill(file, area)
                    if skill.name in skills:
                        raise ValueError(f"Duplicate skill name {skill.name!r} in {file}")
                except (ValueError, OSError, UnicodeDecodeError) as exc:
                    if not is_optional:
                        raise
                    LOGGER.warning("Skipping skill %s: %s", file, exc)
                    continue
                if area is not None and description is not None:
                    areas.setdefault(area, description)
                skills[skill.name] = skill
        return cls(skills, areas)

    def __bool__(self) -> bool:
        return bool(self.skills)

    def catalog_prompt(self) -> str:
        sections = []
        ungrouped = [s for s in self.skills.values() if s.area is None]
        if ungrouped:
            sections.append("\n\n".join(skill.metadata for skill in ungrouped))
        for area, description in self.areas.items():
            members = [s for s in self.skills.values() if s.area == area]
            if members:
                entries = "\n\n".join(skill.metadata for skill in members)
                sections.append(f"## {area}\n{description}\n\n{entries}")
        catalog = "\n\n".join(sections)
        return (
            "# Skills\n"
            "Skills are reusable instructions for specific kinds of work. Only their "
            "names, descriptions and the scenarios they explain are listed here. Before "
            "doing work a skill covers, or answering a question it explains, call "
            "`invoke_skill` with its name to load the full instructions, then follow them."
            f"\n\n<skills>\n{catalog}\n</skills>"
        )

    def has_extra_files(self) -> bool:
        return any(any(f.is_file() and f.name != "SKILL.md" for f in s.path.parent.iterdir())
                   for s in self.skills.values())

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

    def file_tool(self) -> Tool:
        library = self

        @tool(name="read_skill_file")
        def read_skill_file(skill: str, path: str) -> str:
            """Read another file in a skill's folder, such as a reference.md the skill points to.

            Args:
                skill: The skill's name from the skills catalog.
                path: The file's path inside the skill's folder, e.g. reference.md.
            """
            found = library.skills.get(skill.strip())
            if found is None:
                raise ToolError(f"Unknown skill {skill!r}.")
            folder = found.path.parent.resolve()
            target = (folder / path).resolve()
            if folder not in target.parents or not target.is_file():
                raise ToolError(f"{path!r} is not a file in the {skill} skill's folder.")
            if target.stat().st_size > MAX_SKILL_FILE:
                raise ToolError(f"{path!r} is too large to read here.")
            try:
                return target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise ToolError(f"{path!r} is not a text file.") from None

        return read_skill_file
