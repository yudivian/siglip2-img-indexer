import json
import logging
import os
import pickle
import re
from typing import Any, Dict, Generator, List, Set

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"
}

def setup_logging(level: str = "INFO"):
    """Configures the root logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

def find_image_files(directories: List[str]) -> Generator[str, None, None]:
    """
    Recursively finds all image files in the given directories.
    Yields full paths to the image files.
    """
    for directory in directories:
        if not os.path.isdir(directory):
            logging.warning(f"Input path is not a directory, skipping: {directory}")
            continue
        
        logging.info(f"Scanning directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                    yield os.path.join(root, file)

def sanitize_filename(filename: str) -> str:
    """Removes invalid characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def load_index(file_path: str) -> Dict[str, Any]:
    """Loads an index file (JSONL or PKL) into a dictionary keyed by image path."""
    index = {}
    if not os.path.exists(file_path):
        return index
    
    try:
        if file_path.endswith(".jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    index[item["metadata"]["path"]] = item
        elif file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                while True:
                    try:
                        item = pickle.load(f)
                        index[item["metadata"]["path"]] = item
                    except EOFError:
                        break
    except Exception as e:
        logging.error(f"Failed to load index from {file_path}: {e}")
    return index

def save_index(index_data: List[Dict[str, Any]], file_path: str, format: str):
    """Saves the index data to a file in the specified format."""
    try:
        if format == "jsonl":
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in index_data:
                    f.write(json.dumps(item) + "\n")
        elif format == "pkl":
            with open(file_path, 'wb') as f:
                for item in index_data:
                    pickle.dump(item, f)
    except Exception as e:
        logging.error(f"Failed to save index to {file_path}: {e}")

def filter_files_for_reindex(
    all_files_on_disk: Set[str],
    existing_index: Dict[str, Any]
) -> Set[str]:
    """
    Determines which files need to be processed for re-indexing.
    - New files: on disk but not in index.
    - Modified files: modification time on disk is different from index.
    """
    files_to_process = set()
    
    indexed_files = set(existing_index.keys())
    
    new_files = all_files_on_disk - indexed_files
    files_to_process.update(new_files)
    
    for path in indexed_files.intersection(all_files_on_disk):
        try:
            disk_mtime = str(os.path.getmtime(path))
            index_mtime_iso = existing_index[path]["metadata"].get("modification_time")
            if index_mtime_iso:
                pass 
            
            from datetime import datetime, timezone
            
            index_mtime = datetime.fromisoformat(index_mtime_iso).timestamp()

            if abs(float(disk_mtime) - index_mtime) > 1: 
                 files_to_process.add(path)

        except Exception as e:
            logging.warning(f"Could not compare modification time for {path}: {e}, adding to re-index queue.")
            files_to_process.add(path)

    logging.info(f"Found {len(files_to_process)} new or modified files to process.")
    return files_to_process