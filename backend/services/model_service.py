class ModelService:
    CLASSIFICATION_MODELS = [
        {
            "name": "resnet18",
            "display_name": "ResNet-18",
            "params": "11.7M",
            "vram_mb": 800,
            "input_size": 224,
            "speed": "fast",
            "accuracy": "good",
            "description": "Lightweight model, great for small datasets and quick experiments.",
        },
        {
            "name": "resnet50",
            "display_name": "ResNet-50",
            "params": "25.6M",
            "vram_mb": 1500,
            "input_size": 224,
            "speed": "medium",
            "accuracy": "very_good",
            "description": "Strong baseline model with excellent feature extraction.",
        },
        {
            "name": "efficientnet_b0",
            "display_name": "EfficientNet-B0",
            "params": "5.3M",
            "vram_mb": 600,
            "input_size": 224,
            "speed": "fast",
            "accuracy": "very_good",
            "description": "Excellent accuracy/efficiency trade-off. Recommended for most tasks.",
        },
        {
            "name": "efficientnet_b3",
            "display_name": "EfficientNet-B3",
            "params": "12.2M",
            "vram_mb": 1200,
            "input_size": 300,
            "speed": "medium",
            "accuracy": "excellent",
            "description": "Higher accuracy variant of EfficientNet.",
        },
        {
            "name": "mobilenet_v3_small",
            "display_name": "MobileNetV3-Small",
            "params": "2.5M",
            "vram_mb": 400,
            "input_size": 224,
            "speed": "very_fast",
            "accuracy": "fair",
            "description": "Ultra-lightweight, ideal for mobile/edge deployment.",
        },
        {
            "name": "mobilenet_v3_large",
            "display_name": "MobileNetV3-Large",
            "params": "5.5M",
            "vram_mb": 700,
            "input_size": 224,
            "speed": "fast",
            "accuracy": "good",
            "description": "Mobile-friendly model with better accuracy than Small variant.",
        },
        {
            "name": "vgg16",
            "display_name": "VGG-16",
            "params": "138.4M",
            "vram_mb": 2500,
            "input_size": 224,
            "speed": "slow",
            "accuracy": "good",
            "description": "Classic architecture, well-understood. High memory usage.",
        },
        {
            "name": "vit_b_16",
            "display_name": "ViT-B/16",
            "params": "86.6M",
            "vram_mb": 2200,
            "input_size": 224,
            "speed": "medium",
            "accuracy": "excellent",
            "description": "Vision Transformer. Best with larger datasets (1000+ per class).",
        },
        {
            "name": "convnext_tiny",
            "display_name": "ConvNeXt-Tiny",
            "params": "28.6M",
            "vram_mb": 1000,
            "input_size": 224,
            "speed": "medium",
            "accuracy": "excellent",
            "description": "Modern CNN matching ViT performance. Great all-rounder.",
        },
    ]

    DETECTION_MODELS = [
        {
            "name": "yolov8n",
            "display_name": "YOLOv8-Nano",
            "params": "3.2M",
            "vram_mb": 600,
            "input_size": 640,
            "speed": "very_fast",
            "accuracy": "fair",
            "description": "Ultra-fast detection model for real-time applications.",
        },
        {
            "name": "yolov8s",
            "display_name": "YOLOv8-Small",
            "params": "11.2M",
            "vram_mb": 1000,
            "input_size": 640,
            "speed": "fast",
            "accuracy": "good",
            "description": "Good balance of speed and accuracy.",
        },
        {
            "name": "yolov8m",
            "display_name": "YOLOv8-Medium",
            "params": "25.9M",
            "vram_mb": 1800,
            "input_size": 640,
            "speed": "medium",
            "accuracy": "very_good",
            "description": "Higher accuracy detection with moderate speed.",
        },
        {
            "name": "yolov8l",
            "display_name": "YOLOv8-Large",
            "params": "43.7M",
            "vram_mb": 2800,
            "input_size": 640,
            "speed": "slow",
            "accuracy": "excellent",
            "description": "Best accuracy YOLO variant. Requires more VRAM.",
        },
        {
            "name": "fasterrcnn_resnet50_fpn",
            "display_name": "Faster R-CNN (ResNet50-FPN)",
            "params": "41.8M",
            "vram_mb": 2500,
            "input_size": 800,
            "speed": "slow",
            "accuracy": "very_good",
            "description": "Two-stage detector. Good accuracy, slower inference.",
        },
        {
            "name": "ssd300_vgg16",
            "display_name": "SSD300 (VGG16)",
            "params": "35.6M",
            "vram_mb": 2000,
            "input_size": 300,
            "speed": "fast",
            "accuracy": "fair",
            "description": "Single-shot detector. Fast inference, lower accuracy.",
        },
    ]

    SEGMENTATION_MODELS = [
        {
            "name": "deeplabv3_resnet50",
            "display_name": "DeepLabV3 (ResNet50)",
            "params": "42.0M",
            "vram_mb": 2000,
            "input_size": 512,
            "speed": "medium",
            "accuracy": "very_good",
            "description": "Semantic segmentation with atrous convolutions.",
        },
        {
            "name": "deeplabv3_resnet101",
            "display_name": "DeepLabV3 (ResNet101)",
            "params": "61.0M",
            "vram_mb": 3000,
            "input_size": 512,
            "speed": "slow",
            "accuracy": "excellent",
            "description": "Higher capacity DeepLabV3 variant.",
        },
        {
            "name": "unet_resnet34",
            "display_name": "U-Net (ResNet34)",
            "params": "24.4M",
            "vram_mb": 1500,
            "input_size": 256,
            "speed": "medium",
            "accuracy": "very_good",
            "description": "Popular encoder-decoder architecture for segmentation.",
        },
        {
            "name": "unet_efficientnet_b3",
            "display_name": "U-Net (EfficientNet-B3)",
            "params": "14.0M",
            "vram_mb": 1200,
            "input_size": 256,
            "speed": "medium",
            "accuracy": "excellent",
            "description": "U-Net with efficient encoder. Great accuracy/speed balance.",
        },
        {
            "name": "fcn_resnet50",
            "display_name": "FCN (ResNet50)",
            "params": "35.3M",
            "vram_mb": 1800,
            "input_size": 512,
            "speed": "medium",
            "accuracy": "good",
            "description": "Fully Convolutional Network. Classic segmentation approach.",
        },
    ]

    def list_architectures(self, task_type: str) -> list[dict]:
        if task_type == "classification":
            return self.CLASSIFICATION_MODELS
        elif task_type == "detection":
            return self.DETECTION_MODELS
        elif task_type == "segmentation":
            return self.SEGMENTATION_MODELS
        return []

    def recommend(self, task_type: str, num_classes: int, dataset_size: int, gpu_vram_mb: int) -> dict:
        models = self.list_architectures(task_type)
        # Filter by VRAM
        suitable = [m for m in models if m["vram_mb"] <= gpu_vram_mb * 0.8]
        if not suitable:
            suitable = models[:1]  # At least suggest the smallest

        # Score models
        best = suitable[0]
        reasons = []

        if task_type == "classification":
            if dataset_size < 500:
                # Small dataset — use lightweight model with pretrained weights
                for m in suitable:
                    if m["name"] in ("efficientnet_b0", "resnet18", "mobilenet_v3_large"):
                        best = m
                        break
                reasons.append("Small dataset — lightweight pretrained model recommended to avoid overfitting.")
            elif dataset_size < 5000:
                for m in suitable:
                    if m["name"] in ("efficientnet_b0", "resnet50", "convnext_tiny"):
                        best = m
                        break
                reasons.append("Medium dataset — good balance of model capacity and training data.")
            else:
                for m in suitable:
                    if m["name"] in ("convnext_tiny", "efficientnet_b3", "vit_b_16"):
                        best = m
                        break
                reasons.append("Large dataset — larger model can leverage more data.")

        elif task_type == "detection":
            if gpu_vram_mb < 2000:
                best = next((m for m in suitable if "yolov8n" in m["name"] or "yolov8s" in m["name"]), suitable[0])
                reasons.append("Limited VRAM — lightweight YOLO variant recommended.")
            else:
                best = next((m for m in suitable if "yolov8m" in m["name"]), suitable[0])
                reasons.append("Sufficient VRAM — medium YOLO for best accuracy/speed trade-off.")

        elif task_type == "segmentation":
            if gpu_vram_mb < 2000:
                best = next((m for m in suitable if "unet" in m["name"]), suitable[0])
                reasons.append("Limited VRAM — U-Net with efficient encoder recommended.")
            else:
                best = next((m for m in suitable if "deeplabv3" in m["name"]), suitable[0])
                reasons.append("Sufficient VRAM — DeepLabV3 for high-quality segmentation.")

        return {
            "recommended": best,
            "reasons": reasons,
            "all_suitable": suitable,
        }
