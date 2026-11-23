# Research history (single-branch layout)

This repository uses **main** for the public project and current documentation. The old development branches were archived on November 23, 2026.

The original branch names and exact HEAD commit SHAs are recorded in [the complete inventory](archive/branch-heads-2026-11-23.json). Each original commit is preserved in **main's ancestry** through archive-only merge commits, without applying the old experimental trees on top of the shipped runtime.

To view the original source, use the commit permalink or \`git show\` with the archived SHA, for example:

~~~bash
git show b63fa3b088ff608458933531387a3be49ba33049:scripts/gaia_same_task.py
git show a7234b13945efc66ee3b66ce2fccb49997ee09f4:scripts/colab_v8_stage70_remote.py
~~~

For the important November experiment code, copies are also available directly in [archive/2026-11-23/](../archive/2026-11-23/). These are snapshots for audit, **not installed production modules or automatically triggered Actions workflows**.

| Historical work | Archived source |
| --- | --- |
| November paired GAIA100 Base/v7/v8 Step70 | [frozen experiment commit](https://github.com/kolihhan/small-agent-qlora/tree/b63fa3b088ff608458933531387a3be49ba33049) |
| v8 Stage70 training and resume | [training commit](https://github.com/kolihhan/small-agent-qlora/tree/a7234b13945efc66ee3b66ce2fccb49997ee09f4) |
| v8 Stage35 original training code | [training commit](https://github.com/kolihhan/small-agent-qlora/tree/3eac23cca117a41855f4998399f79a5d9d5f4369) |
| Base/v3B/v3D historical evaluation | [GAIA experiment documentation](gaia-v3d-results.md) |
| Earlier controlled study | [v3D report](v3d-results.md) |
| Portfolio documentation changes | [merged PR #117](https://github.com/kolihhan/small-agent-qlora/pull/117) |

The two GAIA100 score tables use different runtime revisions. The November same-run comparison is Base 18/100, v7 25/100 and v8 Step70 29/100. The older Base/v3B/v3D 9/18/18 comparison is separate. No model scores changed during branch cleanup.

**Why retain commit ancestry?** GitHub can garbage-collect commits that are no longer reachable after the only branch pointing at them is deleted. An archive merge with the existing main tree and the historical branch heads as parents preserves those commits without merging their experimental code into main.

GitHub Actions logs and artifact URLs remain separate evidence. Artifacts can expire even though commit history is preserved.
