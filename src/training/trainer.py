"""
SynthMorph Training Loop (Steps 3 & 4)

Orchestrates the full training cycle:
  1. Generate synthetic data (m, f, s_m, s_f) via pipeline
  2. Forward pass through SynthMorphUNet → displacement field phi
  3. Warp moving label map s_m with phi → warped_sm
  4. Compute SoftDiceLoss(warped_sm, s_f) + lambda * GradLoss(phi)
  5. Backward + Adam step
  6. Periodic visualization, checkpointing, LR scheduling

Supports AMP (Mixed Precision) for VRAM efficiency.
"""

import sys
import os
import time
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Local imports resolved relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.network import SynthMorphUNet, SpatialTransformerPure
from src.models.losses import SynthMorphLoss
from src.utils.visualizer import Visualizer
from src.utils.metrics import dice_score, jacobian_determinant
from src.training.checkpointing import CheckpointManager
from src.training.oom_handler import OOMHandler


class Trainer:
    """
    Full SynthMorph training loop handler.

    Args:
        model:           SynthMorphUNet instance
        pipeline:        sm_shapes or sm_brains pipeline with .generate_pair()
        loss_fn:         SynthMorphLoss instance
        optimizer:       Adam optimizer
        device:          torch.device
        config:          training configuration dict
        checkpoint_mgr:  CheckpointManager instance
        visualizer:      Visualizer instance
        use_amp:         whether to use Automatic Mixed Precision (FP16)
        start_iter:      iteration to resume from (0 for fresh start)
    """

    def __init__(
        self,
        model,
        pipeline,
        loss_fn: SynthMorphLoss,
        optimizer,
        device: torch.device,
        config: dict,
        checkpoint_mgr: CheckpointManager,
        visualizer: Visualizer,
        use_amp: bool = True,
        start_iter: int = 0,
    ):
        self.model = model
        self.pipeline = pipeline
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.checkpoint_mgr = checkpoint_mgr
        self.visualizer = visualizer
        self.use_amp = use_amp and torch.cuda.is_available()
        self.start_iter = start_iter

        # Spatial transformer for warping label maps (nearest-neighbor)
        self.label_warper = SpatialTransformerPure(mode='nearest').to(device)

        # Spatial transformer for warping images (bilinear, for visualization)
        self.img_warper = SpatialTransformerPure(mode='bilinear').to(device)

        # Mixed precision scaler (device='cuda' required for torch.amp)
        _amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler(device=_amp_device) if self.use_amp else None

        # LR Scheduler: reduce on plateau (Dice loss) as per paper
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,       # 1e-4 -> 1e-5
            patience=10000,   # steps before reducing
            min_lr=1e-5,
        )

        self._loss_history = []
        self._time_start = None

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------
    def _step(self, oom_handler: OOMHandler = None):
        """
        Execute one training iteration.

        Returns:
            loss_dict: dict with 'loss_total', 'loss_dice', 'loss_grad'
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # ---- Step 1 & 2: Generate synthetic data ----
        with torch.no_grad():
            m, f, s_m, s_f = self.pipeline.generate_pair(batch_size=1)

        m   = m.to(self.device)
        f   = f.to(self.device)
        s_m = s_m.to(self.device)
        s_f = s_f.to(self.device)

        # ---- Step 3: Forward through registration network ----
        _amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.use_amp:
            with autocast(device_type=_amp_device):
                phi = self.model(m, f)            # (B, 3, D, H, W)

                # Warp moving label map with predicted phi
                # NOTE: s_m must be float for SpatialTransformer
                warped_sm = self.label_warper(s_m.float(), phi)

                # ---- Step 4: Compute loss ----
                total_loss, loss_dict = self.loss_fn(phi, warped_sm, s_f)

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            phi = self.model(m, f)
            warped_sm = self.label_warper(s_m.float(), phi)
            total_loss, loss_dict = self.loss_fn(phi, warped_sm, s_f)
            total_loss.backward()
            self.optimizer.step()

        self.scheduler.step(total_loss.item())

        return loss_dict, m, f, s_m, s_f, phi, warped_sm

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, total_iters: int, oom_handler: OOMHandler = None):
        """
        Run the full training loop from start_iter to total_iters.

        Args:
            total_iters: total number of training iterations (paper: 400,000)
            oom_handler: optional OOMHandler for step-level OOM recovery
        """
        self._time_start = time.time()
        log_every = self.config.get('log_every', 100)
        vis_every = self.config.get('vis_every', 1000)
        save_every = self.config.get('save_every', 5000)

        print(f'\n{"="*60}')
        print(f'  SynthMorph Training Started')
        print(f'  Device:       {self.device}')
        print(f'  AMP:          {self.use_amp}')
        print(f'  nb_features:  {self.model.nb_features}')
        print(f'  Start iter:   {self.start_iter}')
        print(f'  Total iters:  {total_iters}')
        print(f'  Params:       {self.model.count_parameters():,}')
        print(f'{"="*60}\n')

        pbar = tqdm(
            total=total_iters, 
            initial=self.start_iter,
            desc="Training",
            dynamic_ncols=True
        )

        for iteration in range(self.start_iter, total_iters):
            try:
                if oom_handler:
                    loss_dict, m, f, s_m, s_f, phi, warped_sm = oom_handler.try_train_step(
                        self._step, oom_handler
                    )
                else:
                    loss_dict, m, f, s_m, s_f, phi, warped_sm = self._step()

            except MemoryError as e:
                pbar.write(str(e))
                raise

            self._loss_history.append(loss_dict['loss_total'])

            # ---- Update Progress Bar ----
            pbar.set_postfix({
                'loss': f"{loss_dict['loss_total']:.4f}",
                'dice': f"{loss_dict['loss_dice']:.4f}"
            })
            pbar.update(1)

            # ---- Visualization (Step 1, 2, 3) ----
            if iteration % vis_every == 0:
                with torch.no_grad():
                    self.visualizer.save_step1(s_m, s_f, iteration,
                                               num_labels=self.config.get('num_labels', 26))
                    self.visualizer.save_step2(m, f, iteration)

                    # Warp intensity image for visualization
                    warped_m_vis = self.img_warper(m, phi.detach())
                    self.visualizer.save_step3(
                        m, f, warped_m_vis, s_f, warped_sm.detach(),
                        phi.detach(), iteration,
                        num_labels=self.config.get('num_labels', 26)
                    )

            # ---- Checkpoint ----
            self.checkpoint_mgr.save(
                self.model, self.optimizer,
                iteration, loss_dict['loss_total']
            )

        pbar.close()

        # Final force-save
        self.checkpoint_mgr.save(
            self.model, self.optimizer,
            total_iters, self._loss_history[-1], force=True
        )
        print('\n✅ Training complete!')
