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
        generator_type: 'baseline' (Variant A) or 'custom' (Variant B)
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

    def _generate_multi_res_deformation(self, batch_size):
        """
        IEEE Paper: "we sample several SVFs v_m ~ N(0, σ_v^2) at resolutions r_v,
        drawing a different σ_v for each to synthesize a more complex deformation...
        The upsampled SVFs are then combined additively"
        """
        svf_total = 0
        for res in config.MULTI_SVF_RES:
            # Draw independent sigma_v ~ U(0, b_v) for each resolution level
            std_v = torch.rand(1).item() * config.MULTI_SVF_MAX_STD
            
            svf_component = svf_generator.generate_svf(
                target_shape=config.TARGET_SHAPE,
                svf_res=res,
                std=std_v,
                device=self.device,
                batch_size=batch_size
            )
            # Additively combine SVF components
            if isinstance(svf_total, int):
                svf_total = svf_component
            else:
                svf_total = svf_total + svf_component
                
        # Integrate combined SVF to smooth diffeomorphic deformation field
        phi = scaling_squaring.integrate_svf(svf_total, steps=config.SCALING_AND_SQUARING_STEPS)
        return phi

    def _synthesize_mri(self, label_map):
        img = gmm_sampler.sample_intensities(
            label_map,
            config.INTENSITY_GMM_MEAN_RANGE,
            config.INTENSITY_GMM_STD_RANGE
        )
        img = blur_pve.apply_anisotropic_blur(img, config.BLUR_MAX_STD)
        img = bias_field.apply_bias_field(img)
        img = augmentation.normalize_min_max(img)
        img = augmentation.apply_gamma(img, config.GAMMA_STD)
        return img

    def generate_pair(self, batch_size=1):
        """
        Generate a synthetic (moving, fixed) MRI pair with shared anatomy.
        """
        # 1. Base anatomy (shared)
        s = self._generate_fn(batch_size, self.device)

        # 2. Independent moving deformation
        phi_m = self._generate_multi_res_deformation(batch_size)
        s_m   = warper.warp_volume(s, phi_m, mode='nearest')

        # 3. Independent fixed deformation
        phi_f = self._generate_multi_res_deformation(batch_size)
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
