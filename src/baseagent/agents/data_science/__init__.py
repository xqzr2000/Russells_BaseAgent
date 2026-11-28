"""The data scientist: notebooks in a persistent Jupyter kernel, Colab runs for GPU work.

* ``agent``: ``DataScienceAgent``, its prompt and tools
* ``kernel`` / ``notebook``: the kernel and the ``.ipynb`` it records (``run_cell``)
* ``probe``: ``inspect_dataset``, a first look at any dataset
* ``kaggle``: Kaggle dataset downloads
* ``colab``: drafts, the CPU smoke run and the Colab T4 run
"""

from baseagent.agents.data_science.agent import DataScienceAgent

__all__ = ["DataScienceAgent"]
