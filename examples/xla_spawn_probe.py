import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr


def _worker(index: int) -> None:
    device = xm.xla_device()
    global_ordinal = xr.global_ordinal() if hasattr(xr, "global_ordinal") else -1
    world_size = (
        xr.global_runtime_device_count()
        if hasattr(xr, "global_runtime_device_count")
        else -1
    )
    print(
        f"[probe] index={index} ordinal={global_ordinal} world_size={world_size} "
        f"xla_device={device} pid={os.getpid()}",
        flush=True,
    )
    t = torch.tensor([index], device=device)
    _ = (t + 1).sum()
    xm.mark_step()


if __name__ == "__main__":
    raw_nprocs = os.environ.get("XLA_PROBE_NPROCS", "auto").strip().lower()
    nprocs = None if raw_nprocs in {"", "auto", "none"} else int(raw_nprocs)
    print(f"[probe] launching xmp.spawn with nprocs={nprocs}", flush=True)
    xmp.spawn(_worker, args=(), nprocs=nprocs, start_method="spawn")
    print("[probe] completed", flush=True)
