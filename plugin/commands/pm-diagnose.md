---
description: Start a bearing fault diagnosis workflow on a vibration signal
argument-hint: "[signal_id or file_path]"
---

Run the bearing-diagnosis skill of this plugin
(`predictive-maintenance:bearing-diagnosis`) on the signal the user named.

Pass the argument, a signal_id or a file path, to the skill as the signal to
diagnose; without an argument, the skill's own signal discovery step applies.
Follow that skill's workflow, relay rules and reporting exactly. This command
adds no step, threshold, verdict or recommendation of its own.
