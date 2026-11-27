---
description: Run a quick vibration health screening on a signal
argument-hint: "[signal_id or file_path]"
---

Run the quick-screening skill of this plugin
(`predictive-maintenance:quick-screening`) on the signal the user named.

Pass the argument, a signal_id or a file path, to the skill as the signal to
screen; without an argument, the skill's own signal selection step applies.
Follow that skill's workflow, relay rules and screening card exactly. This
command adds no step, threshold, verdict or recommendation of its own.
