# TabArena Context-Length Sweeps

This directory contains intermediate long-context PluRel 16B TabArena evaluation artifacts.

Current setup:
- checkpoint: `synthetic-pretrain_rdb_512_size_16b.pt`
- sampling: `use_random_sampling=True`
- `seq_len=2048`: relaunched as 2 shards
- `seq_len=4096`: relaunched as 4 shards

Subdirectories:
- `manifests/`: shard manifests derived from `tabarena_manifest_no_multiclass.csv`
- `sharded/`: in-progress shard CSV outputs
- `old_partial/`: archived partial outputs from the earlier single-GPU runs

Notes:
- The shard CSVs in `sharded/` are snapshots of an in-progress run and may not be complete.
- Final merged CSVs should be produced after all shard runs finish.
