# Week 4: Build the Coding Agent (WIP)

> This chapter is a work in progress.

The repository's `agent.py` demonstrates the smallest useful coding-agent loop:
render tools, generate one structured action, validate it, execute it inside a
workspace boundary, append the observation, and repeat.

Run the reference stack with:

```bash
pdm run agent "inspect this project and summarize its files"
```

Start with read-only tools, then add reviewable writes, step limits, token
budgets, and validation. The serving work from Weeks 2 and 3 should remain below
this loop; the agent consumes a generation interface rather than reaching into
model internals.

{{#include copyright.md}}
