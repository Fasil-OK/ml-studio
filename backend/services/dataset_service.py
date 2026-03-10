import json
import os
import shutil
import zipfile
from pathlib import Path

from fastapi import UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.dataset import Dataset


class DatasetService:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

    async def process_upload(
        self, project_id: str, task_type: str, file: UploadFile, db: AsyncSession
    ) -> Dataset:
        # Save uploaded file
        upload_path = settings.upload_dir / f"{project_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract if ZIP
        extract_dir = settings.dataset_dir / project_id
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        if file.filename.endswith(".zip"):
            with zipfile.ZipFile(upload_path, "r") as zf:
                zf.extractall(extract_dir)
            # If extracted into a single subdirectory, use that
            subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and not any(extract_dir.glob("*.*")):
                inner = subdirs[0]
                for item in inner.iterdir():
                    shutil.move(str(item), str(extract_dir / item.name))
                inner.rmdir()

        # Detect annotation format
        annotation_format = self._detect_format(extract_dir, task_type)

        # Create dataset record
        dataset = Dataset(
            project_id=project_id,
            name=file.filename,
            path=str(extract_dir),
            annotation_format=annotation_format,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        # Run analysis
        dataset = await self.analyze_dataset(dataset, task_type, db)
        return dataset

    async def analyze_dataset(self, dataset: Dataset, task_type: str, db: AsyncSession) -> Dataset:
        dataset_path = Path(dataset.path)

        if task_type == "classification":
            analysis = self._analyze_classification(dataset_path)
        elif task_type == "detection":
            analysis = self._analyze_detection(dataset_path)
        elif task_type == "segmentation":
            analysis = self._analyze_segmentation(dataset_path)
        else:
            analysis = self._analyze_classification(dataset_path)

        dataset.total_images = analysis["total_images"]
        dataset.num_classes = analysis["num_classes"]
        dataset.class_names = json.dumps(analysis["class_names"])
        dataset.class_counts = json.dumps(analysis["class_counts"])
        dataset.image_stats = json.dumps(analysis["image_stats"])
        dataset.quality_issues = json.dumps(analysis["quality_issues"])
        dataset.split_info = json.dumps(analysis.get("split_info", {}))

        await db.commit()
        await db.refresh(dataset)
        return dataset

    def _detect_format(self, path: Path, task_type: str) -> str:
        if task_type == "classification":
            return "imagefolder"
        # Check for COCO format
        for name in ["annotations.json", "_annotations.coco.json"]:
            if (path / name).exists():
                return "coco"
        # Check for YOLO format
        if (path / "labels").exists():
            return "yolo"
        # Check for mask-based segmentation
        if (path / "masks").exists():
            return "masks"
        # Check for VOC format
        if (path / "Annotations").exists():
            return "voc"
        # Check splits
        for split in ["train", "val", "test"]:
            split_dir = path / split
            if split_dir.exists():
                for name in ["_annotations.coco.json", "annotations.json"]:
                    if (split_dir / name).exists():
                        return "coco"
                if (split_dir / "labels").exists():
                    return "yolo"
        return "imagefolder"

    def _analyze_classification(self, path: Path) -> dict:
        class_counts = {}
        widths, heights = [], []
        formats = set()
        quality_issues = []
        corrupt_count = 0

        # Check for train/val/test split structure
        split_dirs = {}
        for split_name in ["train", "val", "test", "training", "validation"]:
            split_path = path / split_name
            if split_path.exists() and split_path.is_dir():
                split_dirs[split_name] = split_path

        # If no splits, treat root as the dataset
        dirs_to_scan = split_dirs.values() if split_dirs else [path]

        small_image_count = 0
        for scan_dir in dirs_to_scan:
            for class_dir in sorted(scan_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                images = [f for f in class_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]
                class_counts[class_name] = class_counts.get(class_name, 0) + len(images)

                # Sample images for stats (up to 50 per class)
                for img_path in images[:50]:
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            widths.append(w)
                            heights.append(h)
                            formats.add(img.format or img_path.suffix)
                            if w < 32 or h < 32:
                                small_image_count += 1
                    except Exception:
                        corrupt_count += 1

        if small_image_count > 0:
            quality_issues.append({
                "type": "small_images",
                "count": small_image_count,
                "message": f"{small_image_count} images are smaller than 32x32 pixels (sampled). Images will be resized during training.",
            })

        total_images = sum(class_counts.values())
        class_names = sorted(class_counts.keys())
        num_classes = len(class_names)

        # Quality checks
        if corrupt_count > 0:
            quality_issues.append({"type": "corrupt_files", "count": corrupt_count})
        if num_classes > 0:
            counts = list(class_counts.values())
            ratio = max(counts) / max(min(counts), 1)
            if ratio > 5:
                quality_issues.append({
                    "type": "class_imbalance",
                    "message": f"Class imbalance ratio: {ratio:.1f}:1",
                    "max_class": max(class_counts, key=class_counts.get),
                    "min_class": min(class_counts, key=class_counts.get),
                })
        if total_images < 100:
            quality_issues.append({"type": "small_dataset", "message": f"Only {total_images} images — high overfitting risk"})
        for cn, count in class_counts.items():
            if count < 10:
                quality_issues.append({"type": "few_samples", "class": cn, "count": count})

        image_stats = {}
        if widths:
            image_stats = {
                "avg_width": round(sum(widths) / len(widths)),
                "avg_height": round(sum(heights) / len(heights)),
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "formats": list(formats),
            }

        split_info = {}
        if split_dirs:
            for name, sp in split_dirs.items():
                count = sum(
                    len([f for f in cd.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS])
                    for cd in sp.iterdir() if cd.is_dir()
                )
                split_info[name] = count

        return {
            "total_images": total_images,
            "num_classes": num_classes,
            "class_names": class_names,
            "class_counts": class_counts,
            "image_stats": image_stats,
            "quality_issues": quality_issues,
            "split_info": split_info,
        }

    def _analyze_detection(self, path: Path) -> dict:
        """Analyze object detection dataset (COCO or YOLO format)."""
        # Try COCO format first
        annotations_file = None
        for name in ["annotations.json", "_annotations.coco.json"]:
            if (path / name).exists():
                annotations_file = path / name
                break
        # Check in splits
        if not annotations_file:
            for split in ["train", "val", "test"]:
                for name in ["_annotations.coco.json", "annotations.json"]:
                    candidate = path / split / name
                    if candidate.exists():
                        annotations_file = candidate
                        break
                if annotations_file:
                    break

        if annotations_file:
            return self._analyze_coco(annotations_file, path)
        return self._analyze_yolo_format(path)

    def _analyze_coco(self, annotations_file: Path, root: Path) -> dict:
        with open(annotations_file) as f:
            coco = json.load(f)

        categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
        class_names = sorted(categories.values())
        num_classes = len(class_names)

        # Count annotations per category
        class_counts = {name: 0 for name in class_names}
        for ann in coco.get("annotations", []):
            cat_name = categories.get(ann["category_id"], "unknown")
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

        total_images = len(coco.get("images", []))
        widths = [img.get("width", 0) for img in coco.get("images", [])]
        heights = [img.get("height", 0) for img in coco.get("images", [])]

        image_stats = {}
        if widths:
            image_stats = {
                "avg_width": round(sum(widths) / len(widths)),
                "avg_height": round(sum(heights) / len(heights)),
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
                "total_annotations": len(coco.get("annotations", [])),
                "avg_annotations_per_image": round(len(coco.get("annotations", [])) / max(total_images, 1), 1),
            }

        quality_issues = []
        if total_images < 50:
            quality_issues.append({"type": "small_dataset", "message": f"Only {total_images} images"})

        return {
            "total_images": total_images,
            "num_classes": num_classes,
            "class_names": class_names,
            "class_counts": class_counts,
            "image_stats": image_stats,
            "quality_issues": quality_issues,
            "split_info": {},
        }

    def _analyze_yolo_format(self, path: Path) -> dict:
        labels_dir = path / "labels"
        images_dir = path / "images"
        if not labels_dir.exists():
            # Try in splits
            for split in ["train", "val", "test"]:
                if (path / split / "labels").exists():
                    labels_dir = path / split / "labels"
                    images_dir = path / split / "images"
                    break

        class_counts = {}
        total_images = 0
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                total_images += 1
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = parts[0]
                            class_counts[f"class_{cls_id}"] = class_counts.get(f"class_{cls_id}", 0) + 1

        class_names = sorted(class_counts.keys())
        return {
            "total_images": total_images,
            "num_classes": len(class_names),
            "class_names": class_names,
            "class_counts": class_counts,
            "image_stats": {},
            "quality_issues": [],
            "split_info": {},
        }

    def _analyze_segmentation(self, path: Path) -> dict:
        """Analyze segmentation dataset (mask-based or COCO format)."""
        masks_dir = path / "masks"
        images_dir = path / "images"

        if not masks_dir.exists():
            # Could be COCO segmentation format
            for name in ["annotations.json", "_annotations.coco.json"]:
                if (path / name).exists():
                    return self._analyze_coco(path / name, path)
            return self._analyze_classification(path)

        total_images = len([f for f in images_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS]) if images_dir.exists() else 0
        total_masks = len([f for f in masks_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTENSIONS])

        # Analyze unique values in masks to find classes
        class_set = set()
        import numpy as np
        for mask_path in list(masks_dir.iterdir())[:50]:
            if mask_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                try:
                    mask = np.array(Image.open(mask_path).convert("L"))
                    class_set.update(mask.flatten().tolist())
                except Exception:
                    pass

        class_names = [f"class_{i}" for i in sorted(class_set)]
        quality_issues = []
        if total_images != total_masks:
            quality_issues.append({
                "type": "mismatch",
                "message": f"Image count ({total_images}) != mask count ({total_masks})",
            })

        return {
            "total_images": total_images,
            "num_classes": len(class_names),
            "class_names": class_names,
            "class_counts": {name: 0 for name in class_names},
            "image_stats": {"total_masks": total_masks},
            "quality_issues": quality_issues,
            "split_info": {},
        }

    def get_sample_images(self, dataset_path: str, class_name: str | None, limit: int) -> list[dict]:
        path = Path(dataset_path)
        samples = []

        if class_name:
            class_dir = path / class_name
            if not class_dir.exists():
                # Check in splits
                for split in ["train", "val", "test"]:
                    candidate = path / split / class_name
                    if candidate.exists():
                        class_dir = candidate
                        break
            if class_dir.exists():
                for img_path in list(class_dir.iterdir())[:limit]:
                    if img_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                        samples.append({
                            "path": f"/storage/datasets/{img_path.relative_to(settings.storage_dir)}",
                            "class": class_name,
                            "filename": img_path.name,
                        })
        else:
            # Get samples from all classes
            for class_dir in sorted(path.iterdir()):
                if not class_dir.is_dir() or class_dir.name.startswith("."):
                    continue
                for img_path in list(class_dir.iterdir())[:max(1, limit // 10)]:
                    if img_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                        samples.append({
                            "path": f"/storage/datasets/{img_path.relative_to(settings.storage_dir)}",
                            "class": class_dir.name,
                            "filename": img_path.name,
                        })
                if len(samples) >= limit:
                    break

        return samples[:limit]
