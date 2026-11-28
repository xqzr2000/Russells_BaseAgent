# Jack (the coordinator)

Jack is the chat room's default agent. He talks with the user, answers simple things
himself and routes everything else. Jack is a Jack Russell Terrier; the persona lives in
the UI, apart from one fixed reply to "who are you?" that never calls the model.

Code: [`src/baseagent/agents/coordinator.py`](../../src/baseagent/agents/coordinator.py).

## How he routes

| The user wants | Jack | Tool |
| --- | --- | --- |
| A quick answer, arithmetic, the date | Answers himself | `calculator`, `current_time` |
| Work a specialist does (EDA on a CSV, a model on a Kaggle dataset) | Delegates a self-contained task, waits for the result, and restates it: key numbers with the test-set size, caveats, downloads by name | `delegate` |
| To learn or play chess | Hands the whole chat to the chess coach | `handoff` |

Adding a specialist needs no change to Jack: any agent with `delegatable = True` appears
in his `delegate` tool, and any with `handoff_target = True` in `handoff`. See
[adding an agent](../adding-an-agent.md).

Jack never runs code or reads data himself. The specialist's own summary stays in a
collapsed tool panel the user may never open, so Jack's reply is what they read.

## Coming next

Jack is meant to represent Russell to recruiters at a public URL: answer questions about
him, match a job description, show the case studies in [`showcase/`](../../showcase/) and
pass a message on. See [the roadmap](../roadmap.md).
