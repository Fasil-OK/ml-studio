import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.architectures.registry import load_model, get_gradcam_target_layer


class Explainer:
    def __init__(self, architecture, num_classes, checkpoint_path, class_names, device="cuda"):
        self.architecture = architecture
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = load_model(architecture, num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def explain(self, image_path: str, method: str, output_dir: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        pred_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"class_{pred_idx}"

        if method == "gradcam":
            self._generate_gradcam(input_tensor, pred_idx, img, output_dir)
        elif method == "integrated_gradients":
            self._generate_ig(input_tensor, pred_idx, img, output_dir)
        elif method == "shap":
            self._generate_shap(input_tensor, pred_idx, img, output_dir)

        return {
            "prediction": pred_class,
            "confidence": round(confidence * 100, 2),
        }

    def _generate_gradcam(self, input_tensor, target_class, original_img, output_dir):
        from captum.attr import LayerGradCam, LayerAttribution

        target_layer = get_gradcam_target_layer(self.model, self.architecture)
        if target_layer is None:
            return

        gradcam = LayerGradCam(self.model, target_layer)
        attributions = gradcam.attribute(input_tensor, target=target_class)
        upsampled = LayerAttribution.interpolate(attributions, (224, 224))

        heatmap = upsampled.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)

        self._save_heatmap(heatmap, original_img, output_dir)

    def _generate_ig(self, input_tensor, target_class, original_img, output_dir):
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(input_tensor, target=target_class, n_steps=50)

        attr = attributions.squeeze().cpu().detach().numpy()
        attr = np.mean(np.abs(attr), axis=0)  # Average over channels
        attr = attr / (attr.max() + 1e-8)

        self._save_heatmap(attr, original_img, output_dir)

    def _generate_shap(self, input_tensor, target_class, original_img, output_dir):
        from captum.attr import GradientShap

        gs = GradientShap(self.model)
        baselines = torch.zeros_like(input_tensor)
        attributions = gs.attribute(input_tensor, baselines=baselines, target=target_class)

        attr = attributions.squeeze().cpu().detach().numpy()
        attr = np.mean(np.abs(attr), axis=0)
        attr = attr / (attr.max() + 1e-8)

        self._save_heatmap(attr, original_img, output_dir)

    def _save_heatmap(self, heatmap: np.ndarray, original_img: Image.Image, output_dir: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from pathlib import Path

        # Save heatmap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(original_img.size)
        heatmap_img.save(Path(output_dir) / "heatmap.png")

        # Save overlay
        original_resized = original_img.resize((224, 224))
        original_arr = np.array(original_resized).astype(float) / 255
        heatmap_resized = cm.jet(heatmap)[:, :, :3]
        overlay = original_arr * 0.5 + heatmap_resized * 0.5
        overlay = np.clip(overlay, 0, 1)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        overlay_img.save(Path(output_dir) / "overlay.png")
