"""CLIP-based visual categorization of video frames."""

from typing import Optional

import numpy as np
import torch

from utils.gpu_utils import clear_gpu_memory, get_device
from utils.logging_utils import get_logger

logger = get_logger("clip_analyzer")


class CLIPAnalyzer:
    """Analyzes video frames using CLIP to determine visual themes."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 device: str = "cuda"):
        """Initialize CLIP analyzer.

        Args:
            model_name: CLIP model name.
            pretrained: Pretrained weights identifier.
            device: Device to use ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device_name = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._device = None

    def load_model(self) -> None:
        """Load the CLIP model and preprocessing pipeline."""
        import open_clip

        self._device = get_device(self.device_name)
        logger.info(f"Loading CLIP model: {self.model_name}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self._device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        logger.info("CLIP model loaded")

    def analyze_frames(
        self,
        frames: list[np.ndarray],
        theme_prompts: dict[str, list[str]],
        top_k: int = 3,
    ) -> dict:
        """Analyze frames against theme prompts.

        Args:
            frames: List of frames as numpy arrays (RGB).
            theme_prompts: Dictionary mapping theme names to lists of text prompts.
            top_k: Number of top themes to return.

        Returns:
            Dictionary with:
                primary_theme: Best matching theme name.
                theme_confidence: Confidence score (0-1).
                all_scores: Dict of all theme scores.
        """
        if not frames or not theme_prompts:
            return {"primary_theme": "Otros", "theme_confidence": 0.0, "all_scores": {}}

        if self.model is None:
            self.load_model()

        from PIL import Image

        # Prepare image tensors (sample representative frames)
        sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]

        image_tensors = []
        for frame in sampled_frames:
            img = Image.fromarray(frame)
            tensor = self.preprocess(img).unsqueeze(0)
            image_tensors.append(tensor)

        images = torch.cat(image_tensors).to(self._device)

        # Prepare text prompts
        all_prompts = []
        theme_indices = {}
        idx = 0
        for theme, prompts in theme_prompts.items():
            theme_indices[theme] = list(range(idx, idx + len(prompts)))
            all_prompts.extend(prompts)
            idx += len(prompts)

        text_tokens = self.tokenizer(all_prompts).to(self._device)

        # Run inference
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity: (num_images x num_prompts)
            similarity = (image_features @ text_features.T).cpu().numpy()

        # Average across all sampled frames
        avg_similarity = similarity.mean(axis=0)

        # Compute theme scores (max across prompts per theme)
        theme_scores = {}
        for theme, indices in theme_indices.items():
            theme_scores[theme] = float(np.max(avg_similarity[indices]))

        # Normalize scores to 0-1 range using softmax-like normalization
        scores_array = np.array(list(theme_scores.values()))
        # Scale for more discriminative scores
        scores_scaled = scores_array * 5
        exp_scores = np.exp(scores_scaled - np.max(scores_scaled))
        normalized = exp_scores / exp_scores.sum()

        theme_names = list(theme_scores.keys())
        normalized_scores = {theme_names[i]: float(normalized[i]) for i in range(len(theme_names))}

        # Find best theme
        best_theme = max(normalized_scores, key=normalized_scores.get)
        best_confidence = normalized_scores[best_theme]

        result = {
            "primary_theme": best_theme,
            "theme_confidence": round(best_confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in sorted(
                normalized_scores.items(), key=lambda x: x[1], reverse=True
            )},
        }

        logger.debug(f"CLIP result: {best_theme} ({best_confidence:.2%})")
        return result

    def unload_model(self) -> None:
        """Unload the CLIP model to free GPU memory."""
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        clear_gpu_memory()
        logger.debug("CLIP model unloaded")
