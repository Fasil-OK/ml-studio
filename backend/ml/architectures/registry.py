import torch
import torch.nn as nn
from torchvision import models


def load_model(architecture: str, num_classes: int, pretrained: bool = True, task_type: str = "classification"):
    if task_type == "classification":
        return _load_classification_model(architecture, num_classes, pretrained)
    elif task_type == "detection":
        return _load_detection_model(architecture, num_classes, pretrained)
    elif task_type == "segmentation":
        return _load_segmentation_model(architecture, num_classes, pretrained)
    raise ValueError(f"Unknown task type: {task_type}")


def _load_classification_model(architecture: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None

    if architecture == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif architecture == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif architecture == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown classification architecture: {architecture}")

    return model


def _load_detection_model(architecture: str, num_classes: int, pretrained: bool):
    if architecture.startswith("yolov8"):
        # YOLOv8 uses ultralytics
        return None  # Handled by DetectionTrainer directly

    weights = "DEFAULT" if pretrained else None
    # +1 for background class in torchvision detection models
    if architecture == "fasterrcnn_resnet50_fpn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    elif architecture == "ssd300_vgg16":
        from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unknown detection architecture: {architecture}")

    return model


def _load_segmentation_model(architecture: str, num_classes: int, pretrained: bool):
    weights = "DEFAULT" if pretrained else None

    if architecture == "deeplabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50(weights=weights)
        model.classifier[4] = nn.Conv2d(256, num_classes, 1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    elif architecture == "deeplabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101(weights=weights)
        model.classifier[4] = nn.Conv2d(256, num_classes, 1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    elif architecture == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50(weights=weights)
        model.classifier[4] = nn.Conv2d(512, num_classes, 1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    elif architecture.startswith("unet"):
        import segmentation_models_pytorch as smp
        encoder = "resnet34" if "resnet34" in architecture else "efficientnet-b3"
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown segmentation architecture: {architecture}")

    return model


def get_gradcam_target_layer(model, architecture: str):
    """Get the target layer for GradCAM based on architecture."""
    if "resnet" in architecture:
        return model.layer4[-1]
    elif "efficientnet" in architecture:
        return model.features[-1]
    elif "vgg" in architecture:
        return model.features[-1]
    elif "mobilenet" in architecture:
        return model.features[-1]
    elif "convnext" in architecture:
        return model.features[-1]
    elif "vit" in architecture:
        return model.encoder.layers[-1]
    return None
