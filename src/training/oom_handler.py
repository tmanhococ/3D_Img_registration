"""
OOM (Out of Memory) Handler for SynthMorph Training

Detects GPU OutOfMemoryError and automatically reduces network capacity
by lowering nb_features through a fallback ladder: 256 → 128 → 64.

If even nb_features=64 causes OOM, the handler stops and provides
actionable guidance to the user (reduce TARGET_SHAPE in config.py).
"""

import gc
import torch


# Fallback ladder for nb_features (descending order)
_FALLBACK_LADDER = [256, 128, 64]


class OOMHandler:
    """
    Automatically handles CUDA Out-Of-Memory errors during model initialization
    and training by reducing the model's nb_features.

    Usage:
        handler = OOMHandler(initial_nb_features=64)

        # On model creation:
        model = handler.try_build_model(build_fn)

        # On training step:
        loss = handler.try_train_step(step_fn)

    After any successful recovery, `handler.nb_features` holds the current value.
    """

    def __init__(self, initial_nb_features: int = 64):
        # Find starting point in the ladder (use the given value or the closest lower one)
        self.nb_features = self._clamp_to_ladder(initial_nb_features)
        self._ladder_idx = _FALLBACK_LADDER.index(self.nb_features)

    def _clamp_to_ladder(self, n: int) -> int:
        """Clamp nb_features to the nearest ladder value (rounded down)."""
        for val in _FALLBACK_LADDER:
            if n >= val:
                return val
        return _FALLBACK_LADDER[-1]

    def _can_fallback(self) -> bool:
        return self._ladder_idx < len(_FALLBACK_LADDER) - 1

    def _do_fallback(self) -> int:
        """Step down to the next nb_features in the ladder."""
        self._ladder_idx += 1
        self.nb_features = _FALLBACK_LADDER[self._ladder_idx]
        return self.nb_features

    @staticmethod
    def _flush_cuda():
        """Free cached GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def try_build_model(self, build_fn, *args, **kwargs):
        """
        Attempt to build a model; if OOM, reduce nb_features and retry.

        Args:
            build_fn: callable(nb_features, *args, **kwargs) → nn.Module
            *args, **kwargs: extra args forwarded to build_fn

        Returns:
            model (nn.Module) built with the highest feasible nb_features
        """
        while True:
            try:
                print(f'[OOMHandler] Building model with nb_features={self.nb_features}...')
                model = build_fn(self.nb_features, *args, **kwargs)
                print(f'[OOMHandler] OK - Model created (nb_features={self.nb_features})')
                return model

            except torch.cuda.OutOfMemoryError:
                self._flush_cuda()
                if self._can_fallback():
                    old = self.nb_features
                    new = self._do_fallback()
                    print(
                        f'\n[OOM] Out of GPU memory!\n'
                        f'    Reducing nb_features: {old} -> {new}\n'
                        f'    Retrying...\n'
                    )
                else:
                    self._flush_cuda()
                    raise MemoryError(
                        '[OOM CRITICAL] Cannot init model even with '
                        f'nb_features={self.nb_features}.\n'
                        'Try reducing TARGET_SHAPE in src/config.py\n'
                        'Example: TARGET_SHAPE = (96, 112, 128)\n'
                    )

    def try_train_step(self, step_fn, *args, **kwargs):
        """
        Attempt a single training step; catch OOM and re-raise with guidance.

        Training-time OOM is harder to recover from (model already built).
        We catch it, flush GPU, and advise the user.

        Args:
            step_fn: callable() → loss
        Returns:
            result of step_fn
        """
        try:
            return step_fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            self._flush_cuda()
            raise MemoryError(
                '[OOM during training step]\n'
                'GPU ran out of memory during training.\n'
                'Options:\n'
                f'  1. Current nb_features={self.nb_features}. Try --nb-features 64\n'
                '  2. Reduce TARGET_SHAPE in src/config.py\n'
                '  3. Check for unreleased tensors in memory\n'
            )
