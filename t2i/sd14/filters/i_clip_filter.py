import os
import torch
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from einops import rearrange, repeat

RESOURCES_ROOT = '/data/yingkai/autosp/sdxl/utils/detection'

def load_model_weights(path: str):
    model_weights = np.load(path)
    return model_weights["weights"], model_weights["biases"]

def clip_process_images(images: torch.Tensor) -> torch.Tensor:
    min_size = min(images.shape[-2:])
    return T.Compose(
        [
            T.CenterCrop(min_size),  # TODO: this might affect the watermark, check this
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )(images)

def predict_proba(X, weights, biases):
    logits = X @ weights.T + biases
    proba = np.where(
        logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits))
    )
    return proba.T

def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1

class DeepFloydDataFiltering(object):
    def __init__(
        self, verbose: bool = False, device: torch.device = torch.device("cuda:6")
    ):
        super().__init__()
        self.verbose = verbose
        self._device = None
        self.clip_model, _ = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()

        self.cpu_w_weights, self.cpu_w_biases = load_model_weights(
            os.path.join(RESOURCES_ROOT, "w_head_v1.npz")
        )
        self.cpu_p_weights, self.cpu_p_biases = load_model_weights(
            os.path.join(RESOURCES_ROOT, "p_head_v1.npz")
        )
        self.w_threshold, self.p_threshold = 0.5, 0.5

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor):
        nsfw = False
        imgs = clip_process_images(images)
        if self._device is None:
            self._device = next(p for p in self.clip_model.parameters()).device
        image_features = self.clip_model.encode_image(imgs.to(self._device))
        image_features = image_features.detach().cpu().numpy().astype(np.float16)
        p_pred = predict_proba(image_features, self.cpu_p_weights, self.cpu_p_biases)
        w_pred = predict_proba(image_features, self.cpu_w_weights, self.cpu_w_biases)
        print(f"p_pred = {p_pred}, w_pred = {w_pred}") if self.verbose else None
        query = p_pred > self.p_threshold
        if query.sum() > 0:
            print(f"Hit for p_threshold: {p_pred}") if self.verbose else None
            batch_size, channels, height, width = images[query].shape
            black_image = torch.zeros((batch_size, channels, height, width), dtype=images[query].dtype).to('cuda:6')
            images[query] = black_image
            nsfw = True
            # images[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(images[query])
        query = w_pred > self.w_threshold
        if query.sum() > 0:
            print(f"Hit for w_threshold: {w_pred}") if self.verbose else None
            batch_size, channels, height, width = images[query].shape
            black_image = torch.zeros((batch_size, channels, height, width), dtype=images[query].dtype).to('cuda:6')
            images[query] = black_image
            nsfw = True
            # images[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(images[query])
        return images, nsfw