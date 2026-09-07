"""Write, read, and verify stage manifests (DESIGN.md §3.2-§3.4).

`manifest.begin(stage, claim_id, run_id)` creates the stage's output
directory and writes an initial manifest; it fails if the directory already
exists — stage directories are write-once (CLAUDE.md), and a re-run is a new
`run_id` with `parent_run_id` set, never an overwrite.

`manifest.read_input(path)` is the only way a stage may read a prior stage's
output. It recomputes the file's sha256, compares it against the hash
recorded in that prior stage's manifest, and raises (naming the path and
both hashes) on any mismatch.

`manifest.complete(...)` / `manifest.halt(..., halt_reason)` finalize a
manifest with `status: complete`, `halted`, or `failed` (there is no fourth
outcome) and append a row to the run's `registry.parquet` (DESIGN §3.4). The
registry append is atomic and idempotent per `(claim_id, run_id, stage)`.
"""
