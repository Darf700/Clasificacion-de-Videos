"""CLIP-based visual categorization for video frames."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from utils.gpu_utils import clear_gpu_memory, get_device
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class CLIPAnalyzer:
    """Analyzes video frames using OpenAI CLIP for theme categorization.

    Compares frame embeddings against text prompts for each theme
    category and returns similarity scores.

    Args:
        model_name: CLIP model variant (e.g., 'ViT-B/32').
        device: Compute device ('cuda' or 'cpu'). Auto-detected if None.
        batch_size: Number of frames to process per GPU batch.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device or get_device()
        self.batch_size = batch_size
        self.model = None
        self.preprocess = None
        self._text_features_cache: Dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        """Load the CLIP model and preprocessing pipeline."""
        try:
            import clip

            self.model, self.preprocess = clip.load(
                self.model_name, device=self.device
            )
            self.model.eval()
            logger.info("CLIP model loaded: %s on %s", self.model_name, self.device)
        except ImportError:
            # Fallback to transformers CLIP
            from transformers import CLIPModel, CLIPProcessor

            model_map = {
                "ViT-B/32": "openai/clip-vit-base-patch32",
                "ViT-B/16": "openai/clip-vit-base-patch16",
                "ViT-L/14": "openai/clip-vit-large-patch14",
            }
            hf_name = model_map.get(self.model_name, "openai/clip-vit-base-patch32")

            self.model = CLIPModel.from_pretrained(hf_name).to(self.device)
            self.preprocess = CLIPProcessor.from_pretrained(hf_name)
            self.model.eval()
            logger.info("CLIP model loaded (HF): %s on %s", hf_name, self.device)

    def analyze_frames(
        self,
        frames: List[np.ndarray],
        theme_prompts: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Analyze video frames against theme prompts.

        Processes a subset of representative frames (evenly spaced)
        and averages the similarity scores across frames.

        Args:
            frames: List of video frames as numpy arrays (BGR).
            theme_prompts: Dictionary mapping theme names to lists of
                text prompts describing each theme.

        Returns:
            Dictionary mapping theme names to average confidence scores
            (0.0 to 1.0).
        """
        if self.model is None:
            self.load_model()

        # Select representative frames (5 evenly spaced)
        num_representative = min(5, len(frames))
        indices = np.linspace(0, len(frames) - 1, num_representative, dtype=int)
        representative = [frames[i] for i in indices]

        # Encode text prompts (cached)
        text_features = self._encode_themes(theme_prompts)

        # Encode frames in batches
        all_scores: Dict[str, List[float]] = {theme: [] for theme in theme_prompts}

        for i in range(0, len(representative), self.batch_size):
            batch = representative[i : i + self.batch_size]
            image_features = self._encode_images(batch)

            if image_features is None:
                continue

            # Compute similarities
            for theme, t_features in text_features.items():
                similarity = self._compute_similarity(image_features, t_features)
                all_scores[theme].extend(similarity)

        # Average scores per theme
        result = {}
        for theme, scores in all_scores.items():
            if scores:
                result[theme] = float(np.mean(scores))
            else:
                result[theme] = 0.0

        return result

    def _encode_themes(
        self, theme_prompts: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Encode theme text prompts into CLIP text features.

        Uses caching to avoid re-encoding the same prompts.

        Args:
            theme_prompts: Theme name -> list of prompt strings.

        Returns:
            Theme name -> averaged text feature tensor.
        """
        result = {}

        for theme, prompts in theme_prompts.items():
            cache_key = f"{theme}:{','.join(prompts)}"
            if cache_key in self._text_features_cache:
                result[theme] = self._text_features_cache[cache_key]
                continue

            try:
                with torch.no_grad():
                    if hasattr(self.model, "encode_text"):
                        # OpenAI CLIP (conditional import: may not be installed)
                        import clip  # noqa: E402

                        tokens = clip.tokenize(prompts).to(self.device)
                        features = self.model.encode_text(tokens)
                    else:
                        # HuggingFace CLIP
                        inputs = self.preprocess(
                            text=prompts, return_tensors="pt", padding=True
                        ).to(self.device)
                        features = self.model.get_text_features(**inputs)

                    features = features / features.norm(dim=-1, keepdim=True)
                    avg_features = features.mean(dim=0, keepdim=True)
                    avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)

                    self._text_features_cache[cache_key] = avg_features
                    result[theme] = avg_features

            except Exception as e:
                logger.warning("Failed to encode theme '%s': %s", theme, e)
                continue

        return result

    def _encode_images(self, frames: List[np.ndarray]) -> Optional[torch.Tensor]:
        """Encode a batch of frames into CLIP image features.

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            Normalized image feature tensor, or None on failure.
        """
        try:
            import cv2

            with torch.no_grad():
                if hasattr(self.model, "encode_image"):
                    # OpenAI CLIP
                    images = []
                    for frame in frames:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        images.append(self.preprocess(pil_img))

                    image_tensor = torch.stack(images).to(self.device)
                    features = self.model.encode_image(image_tensor)
                else:
                    # HuggingFace CLIP
                    pil_images = []
                    for frame in frames:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_images.append(Image.fromarray(rgb))

                    inputs = self.preprocess(
                        images=pil_images, return_tensors="pt"
                    ).to(self.device)
                    features = self.model.get_image_features(**inputs)

                features = features / features.norm(dim=-1, keepdim=True)
                return features

        except Exception as e:
            logger.warning("Failed to encode images: %s", e)
            return None

    def _compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> List[float]:
        """Compute cosine similarity between image and text features.

        Args:
            image_features: (N, D) image embeddings.
            text_features: (1, D) text embedding.

        Returns:
            List of similarity scores (one per image).
        """
        with torch.no_grad():
            similarity = (image_features @ text_features.T).squeeze(-1)
            # Clamp to valid cosine similarity range, then scale to 0-1
            similarity = similarity.clamp(-1.0, 1.0)
            scaled = ((similarity + 1) / 2).cpu().numpy()

            if scaled.ndim == 0:
                return [float(scaled)]
            return scaled.tolist()

    def unload_model(self) -> None:
        """Unload CLIP model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.preprocess = None
            self._text_features_cache.clear()
            clear_gpu_memory()
            logger.info("CLIP model unloaded")
