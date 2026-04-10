import torch
from .. import config
from ..generators.labels import sm_shapes_labels
from ..generators.spatial import svf_generator, scaling_squaring, warper
from ..generators.intensity import gmm_sampler, blur_pve, bias_field, augmentation


class SynthMorphShapesPipeline:
    """
    Pipeline for generating synthetic (m, f, s_m, s_f) pairs.

    Args:
        device: torch device
        generator_type: 'baseline' (Variant A) or 'custom' (Variant B — geometric primitives)
    """

    def __init__(self, device=config.DEVICE, generator_type: str = 'baseline'):
        self.device = device
        self.generator_type = generator_type

        if generator_type == 'baseline':
            from ..generators.labels.sm_shapes_labels import generate_shape_labels
            self._generate_fn = generate_shape_labels
        elif generator_type == 'custom':
            from ..generators.labels.custom_shapes_labels import generate_custom_labels
            self._generate_fn = generate_custom_labels
        else:
            raise ValueError(f"Unknown generator_type '{generator_type}'. "
                             f"Expected 'baseline' or 'custom'.")

    def _synthesize_mri(self, label_map):
        img = gmm_sampler.sample_intensities(
            label_map,
            config.INTENSITY_GMM_MEAN_RANGE,
            config.INTENSITY_GMM_STD_RANGE
        )
        img = blur_pve.apply_anisotropic_blur(img, config.BLUR_STD_RANGE)
        img = bias_field.apply_bias_field(img)
        img = augmentation.normalize_min_max(img)
        img = augmentation.apply_gamma(img, config.GAMMA_RANGE)
        return img

    def generate_pair(self, batch_size=1):
        """
        Generate a synthetic (moving, fixed) MRI pair with shared anatomy.

        Returns:
            m    (B, 1, D, H, W): synthetic moving image
            f    (B, 1, D, H, W): synthetic fixed image
            s_m  (B, 1, D, H, W): moving label map (long)
            s_f  (B, 1, D, H, W): fixed label map (long)
        """
        # 1. Base anatomy (shared)
        s = self._generate_fn(batch_size, self.device)

        # 2. Independent moving deformation
        svf_m = svf_generator.generate_svf(
            config.TARGET_SHAPE, config.SPATIAL_SVF_RES,
            config.SPATIAL_SVF_STD, self.device, batch_size
        )
        phi_m = scaling_squaring.integrate_svf(svf_m, config.SCALING_AND_SQUARING_STEPS)
        s_m   = warper.warp_volume(s, phi_m, mode='nearest')

        # 3. Independent fixed deformation
        svf_f = svf_generator.generate_svf(
            config.TARGET_SHAPE, config.SPATIAL_SVF_RES,
            config.SPATIAL_SVF_STD, self.device, batch_size
        )
        phi_f = scaling_squaring.integrate_svf(svf_f, config.SCALING_AND_SQUARING_STEPS)
        s_f   = warper.warp_volume(s, phi_f, mode='nearest')

        # 4. Independent intensity synthesis
        m = self._synthesize_mri(s_m)
        f = self._synthesize_mri(s_f)

        return m, f, s_m, s_f


if __name__ == "__main__":
    for gtype in ['baseline', 'custom']:
        pipeline = SynthMorphShapesPipeline(generator_type=gtype)
        m, f, s_m, s_f = pipeline.generate_pair(batch_size=1)
        print(f"[{gtype}] m={m.shape} f={f.shape} s_m={s_m.shape} s_f={s_f.shape}")
