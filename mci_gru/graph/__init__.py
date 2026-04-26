"""
Graph construction for MCI-GRU experiments.

Modules:
- builder: Static and dynamic correlation graph building
"""

__all__ = [
    "GraphBuilder",
    "GraphSchedule",
]


def __getattr__(name):
    if name in {"GraphBuilder", "GraphSchedule"}:
        from mci_gru.graph.builder import GraphBuilder, GraphSchedule

        return {"GraphBuilder": GraphBuilder, "GraphSchedule": GraphSchedule}[name]
    raise AttributeError(name)
