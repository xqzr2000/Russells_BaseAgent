"""The BaseAgent framework: the ReAct loop and what every agent shares.

* ``agent``: ``BaseAgent``, the loop, settings and turn results
* ``llm``: model clients (OpenAI, the offline fake, scripted test models)
* ``tools``: the ``@tool`` decorator, tool context, results and errors
* ``events``: the event stream the chat room renders
* ``memory``: the conversation and long-context compaction
* ``skills``: ``SKILL.md`` libraries an agent can load
* ``delegation`` / ``handoff``: how one agent hands work or the whole chat to another
* ``workspace``: each chat's folder (uploads, outputs)
* ``builtin_tools``: small shared tools (calculator, clock, notes)

Specialized agents live in ``baseagent.agents``; ``from baseagent import BaseAgent`` still works.
"""
