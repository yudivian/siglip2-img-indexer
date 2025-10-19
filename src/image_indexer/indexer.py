import json
import logging
import os
import pickle
import torch
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, PreTrainedModel

from . import utils

register_heif_opener()

class ImageIndexer:
    """Handles the entire image indexing workflow with re-indexing and recovery."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device, self.dtype = self._setup_device(self.config["device"])
        logging.info(f"Using device: {self.device} with dtype: {self.dtype}")
        self.model, self.processor = self._load_model()
        if self.model is None or self.processor is None:
            raise RuntimeError("Model initialization failed. Cannot proceed with indexing.")
        
        if self.config.get("create_thumbnails"):
            self.thumbnail_dir = self.config.get("thumbnail_dir", "thumbnails")
            os.makedirs(self.thumbnail_dir, exist_ok=True)
            self.thumbnail_size = tuple(self.config.get("thumbnail_size", (128, 128)))


    def _setup_device(self, device_choice: str) -> Tuple[torch.device, torch.dtype]:
        if device_choice == "auto":
            if torch.cuda.is_available(): device_choice = "cuda"
            elif torch.backends.mps.is_available(): device_choice = "mps"
            else: device_choice = "cpu"
        
        device = torch.device(device_choice)
        dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32
        return device, dtype

    def _load_model(self) -> Tuple[Optional[PreTrainedModel], Optional[Any]]:
        model_name = self.config["model_name"]
        try:
            logging.info(f"Loading model: {model_name}")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            return model, processor
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            return None, None

    def _normalize_embedding(self, embedding: torch.Tensor) -> List[float]:
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding.cpu().to(torch.float32).numpy().tolist()

    def _create_thumbnail(self, image_path: str, img: Image.Image) -> Optional[str]:
        try:
            thumb = img.copy()
            thumb.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            thumb_filename = f"{utils.sanitize_filename(base_name)}.webp"
            thumb_path = os.path.join(self.thumbnail_dir, thumb_filename)
            
            thumb.save(thumb_path, "WEBP", quality=80)
            
            return os.path.abspath(thumb_path)
        except Exception as e:
            logging.warning(f"Could not create thumbnail for {image_path}: {e}")
            return None

    def _extract_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        try:
            with Image.open(image_path) as img:
                stat = os.stat(image_path)
                metadata = {
                    "path": os.path.abspath(image_path),
                    "filename": os.path.basename(image_path),
                    "size_bytes": stat.st_size,
                    "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "width": img.width,
                    "height": img.height,
                    "exif": {}
                }

                if hasattr(img, '_getexif'):
                    exif_data = img._getexif()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            
                            try:
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8', errors='replace').strip('\x00')
                                elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                                    value = str(value)
                                
                                json.dumps(value) 
                                metadata["exif"][str(tag)] = value
                            except (TypeError, OverflowError):
                                metadata["exif"][str(tag)] = str(value)

                if self.config.get("create_thumbnails"):
                    metadata["thumbnail_path"] = self._create_thumbnail(image_path, img)

            return metadata
        except Exception as e:
            logging.warning(f"Could not extract metadata for {image_path}: {e}")
            return None

    def _process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        images_to_process, valid_paths = [], []
        for path in image_paths:
            try:
                images_to_process.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception as e:
                logging.warning(f"Skipping corrupted/unreadable image {path}: {e}")
        if not images_to_process: return []

        with torch.no_grad():
            inputs = self.processor(images=images_to_process, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            normalized_embeddings = self._normalize_embedding(image_features)
        
        batch_results = []
        for i, path in enumerate(valid_paths):
            if metadata := self._extract_metadata(path):
                batch_results.append({"vector": normalized_embeddings[i], "metadata": metadata})
        return batch_results

    def run(self):
        output_file = self.config["output_file"]
        temp_file = f"{output_file}.tmp"
        
        all_image_files = set(utils.find_image_files(self.config["input_dirs"]))
        
        existing_index = {}
        if os.path.exists(output_file) and self.config["reindex"]:
            existing_index = utils.load_index(output_file)
        elif os.path.exists(temp_file):
            logging.info(f"Resuming from temporary file: {temp_file}")
            existing_index = utils.load_index(temp_file)

        files_to_process = utils.filter_files_for_reindex(all_image_files, existing_index)
        
        if not files_to_process and not self.config["reindex"]:
            logging.info("No new images to process.")
            if self.config["reindex"]:
                final_index = [item for item in existing_index.values() if item["metadata"]["path"] in all_image_files]
                utils.save_index(final_index, output_file, self.config["output_format"])
                logging.info(f"Index updated. Removed deleted files. Final count: {len(final_index)}")
            return

        final_index = list(existing_index.values())
        
        batch_size = self.config["batch_size"]
        files_list = list(files_to_process)
        
        with tqdm(total=len(files_list), desc="Indexing images", unit="image") as pbar:
            for i in range(0, len(files_list), batch_size):
                batch_paths = files_list[i:i+batch_size]
                try:
                    batch_results = self._process_batch(batch_paths)
                    final_index.extend(batch_results)
                    utils.save_index(final_index, temp_file, self.config["output_format"])
                    pbar.update(len(batch_paths))
                except Exception as e:
                    logging.error(f"Failed to process batch {i//batch_size + 1}: {e}", exc_info=True)

        if self.config["reindex"]:
            final_index = [item for item in final_index if item["metadata"]["path"] in all_image_files]

        utils.save_index(final_index, output_file, self.config["output_format"])
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        logging.info(f"Indexing complete. Index saved to {output_file} with {len(final_index)} entries.")