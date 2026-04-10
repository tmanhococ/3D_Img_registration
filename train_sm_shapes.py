"""
train_sm_shapes.py — Entry Point for sm-shapes SynthMorph Training

Usage:
    python train_sm_shapes.py [options]

Key Options:
    --generator-type STR Generator type: 'baseline' (Variant A) or 'custom' (Variant B)
    --nb-features INT    U-Net feature maps (local default: 64; cloud: 256)
    --iters INT          Total training iterations (default: 400000)
    --vis-every INT      Save step1/2/3 artifact images every N iters (default: config.VIS_EVERY)
    --save-every INT     Save checkpoint every N iters (default: config.SAVE_EVERY)
    --runs-dir STR       Root dir for run outputs (default: 'runs')
    --drive-dir STR      Persistent backup dir for cloud (e.g. /content/drive/MyDrive/...)
    --resume             Auto-detect and resume last matching run
    --no-amp             Disable Automatic Mixed Precision

Examples:
    # Local dry-run (5 iters, baseline generator)
    python train_sm_shapes.py --iters 5 --vis-every 1 --save-every 5 --nb-features 64

    # Local test with custom generator
    python train_sm_shapes.py --iters 20 --generator-type custom --nb-features 64

    # Kaggle (Variant A, nb-features=256, save to /kaggle/working)
    python train_sm_shapes.py --iters 400000 --nb-features 256 \
        --generator-type baseline --runs-dir /kaggle/working/runs

    # Colab (Variant B, nb-features=256, save to Drive)
    python train_sm_shapes.py --iters 400000 --nb-features 256 \
        --generator-type custom \
        --runs-dir /content/SynthMorph/runs \
        --drive-dir /content/drive/MyDrive/SynthMorph_runs
"""

import sys
import os
import argparse
import datetime
import torch

# Force UTF-8 output on Windows (prevents UnicodeEncodeError for non-ASCII chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ---- Project root on sys.path ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'voxelmorph'))

from src import config as cfg
from src.pipeline.sm_shapes_pipeline import SynthMorphShapesPipeline
from src.models.network import SynthMorphUNet
from src.models.losses import SynthMorphLoss
from src.training.trainer import Trainer
from src.training.checkpointing import (
    CheckpointManager, find_latest_run, make_run_id, configs_compatible
)
from src.training.oom_handler import OOMHandler
from src.utils.visualizer import Visualizer


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SynthMorph sm-shapes registration model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--generator-type', type=str,   default='baseline',
                        choices=['baseline', 'custom'],
                        help='Generator: baseline (Variant A) | custom (Variant B geometric)')
    parser.add_argument('--iters',          type=int,   default=cfg.TOTAL_ITERS)
    parser.add_argument('--nb-features',    type=int,   default=cfg.NB_FEATURES)
    parser.add_argument('--lr',             type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--lambda-reg',     type=float, default=cfg.LAMBDA_REG)
    parser.add_argument('--num-labels',     type=int,   default=cfg.NUM_LABELS)
    parser.add_argument('--vis-every',      type=int,   default=cfg.VIS_EVERY)
    parser.add_argument('--save-every',     type=int,   default=cfg.SAVE_EVERY)
    parser.add_argument('--log-every',      type=int,   default=cfg.LOG_EVERY)
    parser.add_argument('--runs-dir',       type=str,   default='runs')
    parser.add_argument('--drive-dir',      type=str,   default=None,
                        help='Persistent backup dir (e.g. Google Drive path on Colab)')
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--no-amp',         action='store_true')
    return parser.parse_args()


# -----------------------------------------------------------------------
# VRAM Advisory
# -----------------------------------------------------------------------
def print_vram_advisory(nb_features: int):
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'\n[GPU] {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB')
        if vram_gb < 10:
            print(
                f'[Warning] GPU VRAM low ({vram_gb:.1f}GB). '
                f'Using nb_features={nb_features}.\n'
                f'    OOMHandler will auto-reduce nb_features on OOM.\n'
                f'    Or reduce TARGET_SHAPE in src/config.py.\n'
            )
    else:
        print('\n[Device] No GPU found -- using CPU (training will be very slow!).\n')


# -----------------------------------------------------------------------
# Resume logic
# -----------------------------------------------------------------------
def handle_resume(args, current_config: dict, runs_dir: str,
                  prefix: str = 'sm_shapes') -> tuple:
    """Detect existing run and decide resume vs fresh start."""
    latest_run = find_latest_run(runs_dir, prefix=prefix)

    if latest_run is None:
        run_id = make_run_id(prefix)
        return os.path.join(runs_dir, run_id), 0

    saved_config = CheckpointManager.load_config(latest_run)

    # Keys that must match to safely resume
    keys_to_check = ['nb_features', 'num_labels', 'target_shape', 'lambda_reg']
    compatible, mismatches = configs_compatible(saved_config, current_config, keys_to_check)

    saved_iter = saved_config.get('current_iter', 0)
    print(f'\n[Resume] Found existing run: {latest_run}')
    print(f'[Resume] Previous run stopped at iter={saved_iter}')

    if not compatible:
        print('\n[Warning] Current config DOES NOT MATCH saved run:')
        for k, sv, cv in mismatches:
            print(f'    {k}: saved={sv!r} -> current={cv!r}')

    if args.resume:
        decision = 'r'
    else:
        print('\nBạn muốn:')
        print('  [R] Resume từ checkpoint cuối')
        print('  [N] Bắt đầu training mới')
        decision = input('Chọn (R/N): ').strip().lower()

    if decision == 'r':
        print(f'[Resume] Tiếp tục từ iter {saved_iter}')
        return latest_run, saved_iter
    else:
        run_id = make_run_id(prefix)
        print(f'[Resume] Bắt đầu run mới: {run_id}')
        return os.path.join(runs_dir, run_id), 0


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    args = parse_args()
    device = cfg.DEVICE

    print_vram_advisory(args.nb_features)

    # ---- Build config dict (used for JSON logging) ----
    current_config = {
        'run_id':           None,
        'model':            'SynthMorphUNet',
        'pipeline':         'sm-shapes',
        'generator_type':   args.generator_type,
        'nb_features':      args.nb_features,
        'integration_steps': 5,
        'lr':               args.lr,
        'lambda_reg':       args.lambda_reg,
        'num_labels':       args.num_labels,
        'target_shape':     list(cfg.TARGET_SHAPE),
        'total_iters':      args.iters,
        'use_amp':          not args.no_amp and torch.cuda.is_available(),
        'timestamp_start':  datetime.datetime.now().isoformat(),
    }

    # ---- Resume / fresh start ----
    runs_dir = os.path.join(PROJECT_ROOT, args.runs_dir) \
               if not os.path.isabs(args.runs_dir) else args.runs_dir
    run_prefix = f'sm_shapes_{args.generator_type}'
    run_dir, start_iter = handle_resume(args, current_config, runs_dir, prefix=run_prefix)

    run_id = os.path.basename(run_dir)
    current_config['run_id'] = run_id
    vis_dir = os.path.join(run_dir, 'vis')

    # Drive dir (absolute if provided; relative to run_dir if using shorthand)
    drive_dir = args.drive_dir
    if drive_dir and not os.path.isabs(drive_dir):
        drive_dir = os.path.join(run_dir, drive_dir)
    if drive_dir:
        # Mirror run folder inside drive_dir
        drive_dir = os.path.join(drive_dir, run_id)

    print(f'\n[Run] ID:            {run_id}')
    print(f'[Run] Generator:     {args.generator_type}')
    print(f'[Run] Dir:           {run_dir}')
    print(f'[Run] Vis dir:       {vis_dir}')
    print(f'[Run] AMP:           {current_config["use_amp"]}')
    if drive_dir:
        print(f'[Run] Drive backup:  {drive_dir}')

    # ---- Initialize components ----
    oom_handler = OOMHandler(initial_nb_features=args.nb_features)

    # Build model (with OOM fallback)
    def build_model(nb_features):
        m = SynthMorphUNet(nb_features=nb_features).to(device)
        return m

    model = oom_handler.try_build_model(build_model)
    current_config['nb_features'] = oom_handler.nb_features  # might have been reduced

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pipeline = SynthMorphShapesPipeline(device=device, generator_type=args.generator_type)
    loss_fn  = SynthMorphLoss(
        num_labels=args.num_labels,
        lambda_reg=args.lambda_reg
    )

    checkpoint_mgr = CheckpointManager(
        run_dir=run_dir,
        save_every=args.save_every,
        config=current_config,
        drive_dir=drive_dir,
    )

    visualizer = Visualizer(
        save_dir=vis_dir,
        save_every=args.vis_every,
    )

    # ---- Resume: load weights if resuming ----
    if start_iter > 0:
        loaded_iter = checkpoint_mgr.load_latest_periodic(model, optimizer)
        if loaded_iter == 0:
            print('[Resume] No checkpoint file found; starting from iter 0.')
            start_iter = 0

    # ---- Start trainer ----
    trainer = Trainer(
        model=model,
        pipeline=pipeline,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        config={
            **current_config,
            'log_every':  args.log_every,
            'vis_every':  args.vis_every,
            'save_every': args.save_every,
        },
        checkpoint_mgr=checkpoint_mgr,
        visualizer=visualizer,
        use_amp=current_config['use_amp'],
        start_iter=start_iter,
    )

    trainer.train(total_iters=args.iters, oom_handler=oom_handler)


if __name__ == '__main__':
    main()
