# Isolated admission compatibility evidence

The executed script adds narrow support for this task's original legacy preparation receipt while retaining the official pool's admission, resource, ownership, storage, and dispatch checks. It changes only the submitting process. The proof records that shared files were unchanged and the runtime identity matched the original GPU job.

`pool_legacy_admission.py` is an exact copy of the executed script; its SHA256 matches `compatibility_script_sha256` in the proof. Its recorded absolute local pool directory remains part of that original script. `test_legacy_pool_admission.py` contains the eight final synthetic fixtures. The initial red log records seven fixtures at that earlier development stage; the final green log records eight passing fixtures. These results are retained rather than rerun during publication preparation.

`legacy-admission-proof.redacted.json` is an explicitly redacted public copy. One private IPv4 address was replaced by `<private-ip-redacted>` inside `current_task.preparation_errors.h20-03`. That field is an old SSH diagnostic, not a model response or score. The original proof remains unchanged in the task evidence; all other public proof fields remain unchanged.

The `full-pair-v10-compat-submit*` files describe the first v10 submission. Its allocation used different GPUs, and its fixed-input check rejected combining those GPUs with the retained v9 output; the original status and exit records are in `../raw/fullpair-v10-a80de6bd/`. The `full-pair-v10-pair2-compat-submit*` files describe the subsequent fresh complete-pair submission. A submission record does not establish model completion.
