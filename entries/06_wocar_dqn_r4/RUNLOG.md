# Rev 4 run log

Progress file of the rev 4 training (`docs/rev4x_runbook.md`). Newest entries are at the bottom.
The orchestrator, `evaluate.py` and `gate.py` append here by themselves; the training agent adds a line when it
starts a step or sees a problem: `python -c "import rev4; rev4.log('...')"`.
Live training progress is in `PROGRESS.md`; the gates' decisions are in `rev4_params.json`.
