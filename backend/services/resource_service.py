import psutil
import torch


class ResourceService:
    def get_system_resources(self) -> dict:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        ram = psutil.virtual_memory()

        resources = {
            "cpu": {
                "count": cpu_count,
                "frequency_mhz": round(cpu_freq.current) if cpu_freq else None,
                "usage_percent": psutil.cpu_percent(interval=0.1),
            },
            "ram": {
                "total_gb": round(ram.total / 1e9, 1),
                "available_gb": round(ram.available / 1e9, 1),
                "used_gb": round(ram.used / 1e9, 1),
                "percent": ram.percent,
            },
            "gpu": None,
        }

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            resources["gpu"] = {
                "name": gpu_props.name,
                "vram_total_mb": round(gpu_props.total_mem / 1e6),
                "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1e6),
                "vram_free_mb": round((gpu_props.total_mem - torch.cuda.memory_allocated(0)) / 1e6),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            }
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                resources["gpu"]["utilization_percent"] = util.gpu
                resources["gpu"]["memory_utilization_percent"] = util.memory
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                resources["gpu"]["temperature_c"] = temp
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return resources

    def recommend(self, architecture: str, batch_size: int) -> dict:
        resources = self.get_system_resources()
        gpu = resources.get("gpu")

        # Estimate VRAM needs (rough heuristic)
        model_vram_estimates = {
            "resnet18": 800, "resnet50": 1500, "efficientnet_b0": 600,
            "efficientnet_b3": 1200, "mobilenet_v3_small": 400,
            "mobilenet_v3_large": 700, "vgg16": 2500, "vit_b_16": 2200,
            "convnext_tiny": 1000, "yolov8n": 600, "yolov8s": 1000,
            "yolov8m": 1800, "yolov8l": 2800, "fasterrcnn_resnet50_fpn": 2500,
            "deeplabv3_resnet50": 2000, "unet_resnet34": 1500,
        }
        model_vram = model_vram_estimates.get(architecture, 1500)
        # Rough: batch_size affects VRAM linearly
        estimated_vram = model_vram + batch_size * 10  # ~10MB per image in batch

        recommendations = {
            "estimated_vram_mb": estimated_vram,
            "recommended_device": "cuda" if gpu else "cpu",
            "mixed_precision": False,
            "recommended_batch_size": batch_size,
            "recommended_workers": min(4, psutil.cpu_count(logical=True) - 1),
            "warnings": [],
        }

        if gpu:
            vram_free = gpu["vram_free_mb"]
            if estimated_vram > vram_free:
                # Suggest fixes
                recommendations["warnings"].append(
                    f"Estimated VRAM ({estimated_vram}MB) exceeds available ({vram_free}MB)"
                )
                recommendations["mixed_precision"] = True
                # Reduce batch size
                while estimated_vram > vram_free and batch_size > 4:
                    batch_size //= 2
                    estimated_vram = model_vram + batch_size * 10
                recommendations["recommended_batch_size"] = batch_size
        else:
            recommendations["warnings"].append("No GPU detected. Training will be slow on CPU.")

        return recommendations
