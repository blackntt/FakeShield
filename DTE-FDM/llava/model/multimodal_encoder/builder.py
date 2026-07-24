import os
from .clip_encoder import CLIPVisionTower
from .noiseprint_encoder import NoiseprintProjector

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_noiseprint_projector(noiseprint_cfg, **kwargs):
    weights_path = getattr(noiseprint_cfg, 'noiseprint_projector_path', None)
    weights_path = weights_path or "weights/np_plus_plus.pt"
    return NoiseprintProjector(weights_path=weights_path, mm_hidden_size=getattr(noiseprint_cfg, 'noiseprint_projector_mm_hidden_size', 4096), num_tokens=getattr(noiseprint_cfg, 'noiseprint_projector_num_tokens', 64))