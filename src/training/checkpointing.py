"""
Smart Checkpointing for SynthMorph Training

Features:
  - Saves model every N iterations AND whenever best loss is achieved
  - Optional copy to persistent Drive directory (for Google Colab / cloud)
  - Writes/updates config.json metadata file alongside each checkpoint
  - Supports intelligent resume: validates config match before loading
  - Drive sync: automatically copies best + periodic checkpoints to --drive-dir

Directory structure:
  runs/
  └── {run_id}/
      ├── config.json           ← full run config + current progress
      ├── checkpoint_best.pth   ← best model weights (lowest loss)
      └── checkpoint_{iter}.pth ← periodic snapshots

Drive mirror (optional):
  /content/drive/MyDrive/SynthMorph_runs/{run_id}/  ← same structure
"""

import os
import json
import shutil
import datetime
import torch


class CheckpointManager:
    """
    Manages model checkpointing, run config, and optional Drive sync.

    Args:
        run_dir (str):      Local run directory (e.g., 'runs/sm_shapes_...')
        save_every (int):   Periodic save interval in iterations (default from config)
        config (dict):      Training configuration dictionary
        drive_dir (str):    Optional persistent backup directory (Google Drive / any path)
    """

    def __init__(
        self,
        run_dir: str,
        save_every: int = 20,
        config: dict = None,
        drive_dir: str = None,
    ):
        self.run_dir    = run_dir
        self.save_every = save_every
        self.config     = config or {}
        self.drive_dir  = drive_dir
        self.best_loss  = float('inf')
        self._current_iter = 0

        os.makedirs(run_dir, exist_ok=True)
        if drive_dir:
            os.makedirs(drive_dir, exist_ok=True)
            print(f'[Checkpoint] Drive sync enabled → {drive_dir}')

        # Write initial config JSON if not yet present
        self.config_path = os.path.join(run_dir, 'config.json')
        if not os.path.exists(self.config_path):
            self._write_config()

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------
    def _write_config(self):
        """Write current training state to config.json (local + drive)."""
        state = {
            **self.config,
            'current_iter': self._current_iter,
            'best_loss':    self.best_loss if self.best_loss < float('inf') else None,
            'last_updated': datetime.datetime.now().isoformat(),
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        # Mirror config to Drive
        if self.drive_dir:
            shutil.copy2(self.config_path, os.path.join(self.drive_dir, 'config.json'))

    @staticmethod
    def load_config(run_dir: str) -> dict:
        """Load config.json from an existing run directory."""
        path = os.path.join(run_dir, 'config.json')
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             iteration: int, loss: float, force: bool = False):
        """
        Conditionally save checkpoint.

        Logic:
          1. Always save if new best loss achieved → 'checkpoint_best.pth'
          2. Save periodicaly every save_every iters → 'checkpoint_{iter:07d}.pth'
          3. Copy both to drive_dir if set

        Args:
            model:     nn.Module to save
            optimizer: optimizer (state saved for resume)
            iteration: current iteration number
            loss:      current training loss scalar
            force:     if True, force save regardless of schedule
        """
        self._current_iter = iteration
        saved_any = False

        # Best model
        if loss < self.best_loss:
            self.best_loss = loss
            self._save_checkpoint(model, optimizer, iteration, loss, tag='best')
            saved_any = True

        # Periodic save
        if force or (self.save_every > 0 and iteration % self.save_every == 0 and iteration > 0):
            self._save_checkpoint(model, optimizer, iteration, loss,
                                  tag=f'{iteration:07d}')
            saved_any = True

        if saved_any:
            self._write_config()

    def _save_checkpoint(self, model, optimizer, iteration, loss, tag: str):
        """Save a .pth file with full training state."""
        filename = f'checkpoint_{tag}.pth'
        local_path = os.path.join(self.run_dir, filename)

        payload = {
            'iteration':             iteration,
            'loss':                  loss,
            'best_loss':             self.best_loss,
            'model_state_dict':      model.state_dict(),
            'optimizer_state_dict':  optimizer.state_dict(),
        }
        torch.save(payload, local_path)
        print(f'[Checkpoint] Saved  --> {local_path}  (loss={loss:.5f})')

        # Mirror to Drive
        if self.drive_dir:
            drive_path = os.path.join(self.drive_dir, filename)
            shutil.copy2(local_path, drive_path)
            print(f'[Checkpoint] Drive  --> {drive_path}')

    # ------------------------------------------------------------------
    # Load / Resume
    # ------------------------------------------------------------------
    def load_best(self, model, optimizer=None) -> int:
        """Load best checkpoint weights. Returns iteration number."""
        return self._load_checkpoint('best', model, optimizer)

    def load_latest_periodic(self, model, optimizer=None) -> int:
        """Load most recent periodic checkpoint (highest iter number)."""
        checkpoints = [
            f for f in os.listdir(self.run_dir)
            if f.startswith('checkpoint_') and f.endswith('.pth')
            and f != 'checkpoint_best.pth'
        ]
        if not checkpoints:
            print('[Checkpoint] No periodic checkpoint found.')
            return 0

        def _iter_from_name(n):
            try:
                return int(n.replace('checkpoint_', '').replace('.pth', ''))
            except ValueError:
                return -1

        latest = max(checkpoints, key=_iter_from_name)
        tag    = latest.replace('checkpoint_', '').replace('.pth', '')
        return self._load_checkpoint(tag, model, optimizer)

    def _load_checkpoint(self, tag: str, model, optimizer=None) -> int:
        path = os.path.join(self.run_dir, f'checkpoint_{tag}.pth')
        if not os.path.exists(path):
            print(f'[Checkpoint] File not found: {path}')
            return 0
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.best_loss = ckpt.get('best_loss', float('inf'))
        iteration      = ckpt.get('iteration', 0)
        print(f'[Checkpoint] Resumed from iter={iteration}, loss={ckpt.get("loss", "?")}' if isinstance(ckpt.get('loss'), float) else f'[Checkpoint] Resumed from iter={iteration}')
        return iteration


# ------------------------------------------------------------------
# Run directory helpers
# ------------------------------------------------------------------

def find_latest_run(runs_dir: str, prefix: str) -> str | None:
    """Find the most recently modified run directory matching a prefix."""
    if not os.path.isdir(runs_dir):
        return None
    candidates = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if d.startswith(prefix) and os.path.isdir(os.path.join(runs_dir, d))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def make_run_id(prefix: str) -> str:
    """Generate a timestamped run ID."""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{ts}'


def configs_compatible(saved: dict, current: dict, keys_to_check: list) -> tuple[bool, list]:
    """Compare two configs for key compatibility (resume validation)."""
    mismatches = []
    for k in keys_to_check:
        sv = saved.get(k)
        cv = current.get(k)
        if sv != cv:
            mismatches.append((k, sv, cv))
    return len(mismatches) == 0, mismatches
