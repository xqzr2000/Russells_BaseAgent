"""The skill loader: area folders, several roots, optional roots and the catalog."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from baseagent.agents.data_science import KNOWLEDGE_SKILLS, DataScienceAgent
from baseagent.llm import ScriptedLLM
from baseagent.skills import SkillLibrary
from baseagent.tools import ToolError

ROOT = Path(__file__).resolve().parents[1]


def skill(folder: Path, name: str, description: str, body: str = "Do the thing.") -> Path:
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    return folder


def area(folder: Path, title: str, description: str) -> Path:
    folder.mkdir(parents=True)
    (folder / "GROUP.md").write_text(f"---\ntitle: {title}\ndescription: {description}\n---\n")
    return folder


SPLITTING = """# Data splitting

## Scenarios it explains
- **"Validation looked great, but the test score is much lower."** Tuned on test.
- **"Scores jump between runs."** Small sets.

## Pitfalls
- **Not a scenario.** Bold text outside the section isn't one.
"""


def test_area_folders_group_the_catalog_and_list_scenarios(tmp_path):
    evaluation = area(tmp_path / "skills" / "evaluation", "Model evaluation", "Estimating performance.")
    skill(evaluation / "data-splitting", "data-splitting", "Use when splitting data.", SPLITTING)
    skill(tmp_path / "skills" / "plain", "plain", "A skill outside any area.")

    library = SkillLibrary.load(tmp_path / "skills")
    assert set(library.skills) == {"data-splitting", "plain"}
    assert library.skills["data-splitting"].area == "Model evaluation"
    assert library.skills["data-splitting"].scenarios == (
        '"Validation looked great, but the test score is much lower"; "Scores jump between runs"')
    catalog = library.catalog_prompt()
    assert "## Model evaluation\nEstimating performance." in catalog
    assert catalog.index("name: plain") < catalog.index("## Model evaluation") < catalog.index("name: data-splitting")
    assert "Not a scenario" not in catalog
    assert "answering a question it explains" in catalog


def test_long_scenario_lists_are_shortened_for_the_catalog(tmp_path):
    body = "## Scenarios it explains\n" + "".join(f'- **"Symptom number {i} is long enough."** Why.\n'
                                                  for i in range(30))
    skill(tmp_path / "s" / "many", "many", "Use when many.", body)
    scenarios = SkillLibrary.load(tmp_path / "s").skills["many"].scenarios
    assert len(scenarios) <= 245 and scenarios.endswith("; …")


def test_several_roots_load_together_and_names_must_be_unique(tmp_path):
    skill(tmp_path / "a" / "one", "one", "First.")
    skill(tmp_path / "b" / "two", "two", "Second.")
    assert set(SkillLibrary.load(tmp_path / "a", optional=[tmp_path / "b"]).skills) == {"one", "two"}
    skill(tmp_path / "c" / "again", "one", "A duplicate name.")
    with pytest.raises(ValueError, match="Duplicate skill name 'one'"):
        SkillLibrary.load(tmp_path / "a", tmp_path / "c")


def test_optional_roots_may_be_missing_or_broken(tmp_path, caplog):
    skill(tmp_path / "required" / "one", "one", "First.")
    broken = area(tmp_path / "optional" / "evaluation", "Evaluation", "Scores.")
    skill(broken / "good", "good", "Loads.")
    (broken / "bad").mkdir()
    (broken / "bad" / "SKILL.md").write_text("no frontmatter here")

    with caplog.at_level(logging.WARNING):
        library = SkillLibrary.load(tmp_path / "required",
                                    optional=[tmp_path / "missing", tmp_path / "optional"])
    assert set(library.skills) == {"one", "good"}
    assert "Skipping skill" in caplog.text
    with pytest.raises(ValueError, match="missing YAML frontmatter"):  # required roots stay strict
        SkillLibrary.load(tmp_path / "optional")
    with pytest.raises(ValueError, match="not a directory"):
        SkillLibrary.load(tmp_path / "missing")


def test_read_skill_file_stays_inside_the_skill_folder(tmp_path):
    folder = skill(tmp_path / "s" / "pca", "pca", "Use when reducing.")
    (folder / "reference.md").write_text("Longer notes.")
    (tmp_path / "s" / "secret.txt").write_text("not part of the skill")
    library = SkillLibrary.load(tmp_path / "s")
    assert library.has_extra_files()
    read = library.file_tool().handler
    assert read("pca", "reference.md") == "Longer notes."
    for path in ("../secret.txt", "/etc/passwd", "SKILL.md/.."):
        with pytest.raises(ToolError, match="not a file"):
            read("pca", path)
    with pytest.raises(ToolError, match="Unknown skill"):
        read("nope", "reference.md")


@pytest.mark.skipif(not KNOWLEDGE_SKILLS.is_dir(), reason="knowledge/ submodule not checked out")
def test_the_knowledge_repo_layout_loads():
    library = SkillLibrary.load(optional=[KNOWLEDGE_SKILLS])
    assert "pca" in library.skills
    assert library.skills["pca"].area == "Unsupervised learning"
    assert '"PC1 explains 99% of the variance"' in library.skills["pca"].scenarios
    assert "## Unsupervised learning" in library.catalog_prompt()


def test_data_scientist_loads_knowledge_skills_and_works_without_them(tmp_path, monkeypatch):
    knowledge = area(tmp_path / "knowledge" / "evaluation", "Model evaluation", "Estimating performance.")
    skill(knowledge / "data-splitting", "data-splitting", "Use when splitting data.", SPLITTING)

    def tools_and_prompt(root: Path):
        monkeypatch.setattr(DataScienceAgent, "optional_skills_dirs", (root,))
        agent = DataScienceAgent(ScriptedLLM([]), DataScienceAgent.make_settings(), session_id="t",
                                 log_dir=tmp_path / "runs")
        agent.setup = lambda: asyncio.sleep(0)  # no kernel needed to check tools and prompt
        asyncio.run(agent.ensure_ready())
        return set(agent.tools), agent.full_system_prompt()

    tools, prompt = tools_and_prompt(tmp_path / "knowledge")
    assert "invoke_skill" in tools and "name: data-splitting" in prompt
    assert "Data-science basics" in prompt and "From the data-splitting skill" in prompt
    tools, prompt = tools_and_prompt(tmp_path / "no-knowledge")  # public users: no submodule
    assert "invoke_skill" not in tools and "<skills>" not in prompt
    assert "Data-science basics" in prompt
