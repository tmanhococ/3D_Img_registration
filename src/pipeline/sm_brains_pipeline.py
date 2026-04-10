import torch
from .. import config
from ..generators.labels import sm_brains_labels
from ..generators.spatial import svf_generator, scaling_squaring, warper
from ..generators.intensity import gmm_sampler, blur_pve, bias_field, augmentation

class SynthMorphBrainsPipeline:
    def __init__(self, device=config.DEVICE):
        self.device = device
        
    def _synthesize_mri(self, label_map):
        # Sample intensities (GMM)
        img = gmm_sampler.sample_intensities(
            label_map, 
            config.INTENSITY_GMM_MEAN_RANGE, 
            config.INTENSITY_GMM_STD_RANGE
        )
        # Apply Partial Volume Effect (Blur)
        img = blur_pve.apply_anisotropic_blur(img, config.BLUR_STD_RANGE)
        # Apply Bias Field
        img = bias_field.apply_bias_field(img)
        # Normalize and Gamma
        img = augmentation.normalize_min_max(img)
        img = augmentation.apply_gamma(img, config.GAMMA_RANGE)
        return img

    def generate_pair(self, batch_size=1):
        """
        Generates a synthetic moving and fixed MR-like image pair starting from 
        real Buckner40 anatomical maps (sm-brains).
        
        Returns:
            m (torch.Tensor): Synthetic moving image, shape (B, 1, D, H, W)
            f (torch.Tensor): Synthetic fixed image, shape (B, 1, D, H, W)
            s_m, s_f: Optional returning of generated anatomical labels for metric evaluation.
        """
        # 1. Create Base Anatomy (s) from Buckner40 Sample
        # Since sm_brains_labels currently fetches 1 randomly, we need loop for batch
        batch_s = []
        for _ in range(batch_size):
            s_item = sm_brains_labels.get_random_aseg(self.device)
            batch_s.append(s_item)
        s = torch.cat(batch_s, dim=0)
        
        # 2. Sample Independent Deformations (Moving / Fixed)
        # Moving SVF
        svf_m = svf_generator.generate_svf(
            config.TARGET_SHAPE, config.SPATIAL_SVF_RES, config.SPATIAL_SVF_STD, 
            self.device, batch_size
        )
        phi_m = scaling_squaring.integrate_svf(svf_m, config.SCALING_AND_SQUARING_STEPS)
        s_m = warper.warp_volume(s, phi_m, mode='nearest')
        
        # Fixed SVF
        svf_f = svf_generator.generate_svf(
            config.TARGET_SHAPE, config.SPATIAL_SVF_RES, config.SPATIAL_SVF_STD, 
            self.device, batch_size
        )
        phi_f = scaling_squaring.integrate_svf(svf_f, config.SCALING_AND_SQUARING_STEPS)
        s_f = warper.warp_volume(s, phi_f, mode='nearest')
        
        # 3. Independent appearance synthesis
        m = self._synthesize_mri(s_m)
        f = self._synthesize_mri(s_f)
        
        return m, f, s_m, s_f

if __name__ == "__main__":
    pipeline = SynthMorphBrainsPipeline()
    try:
        m, f, s_m, s_f = pipeline.generate_pair(batch_size=1)
        print("sm-brains pipeline executed successfully.")
        print("Moving Shape:", m.shape)
        print("Fixed Shape:", f.shape)
    except Exception as e:
        print(f"Buckner Data error: {e}")
