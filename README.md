# Siglip2 Image Indexer v1.0.1

Image Indexer is a powerful and flexible command-line tool designed to create a vector index from your local image directories. It uses state-of-the-art vision models from the Hugging Face library to generate embeddings for each image, storing them in an efficient format for use in similarity search, image retrieval, and other machine learning applications.

This tool is powered by **Siglip 2**, a powerful vision model from Google, to generate high-quality embeddings for your images.

-----

## Key Features ✨

  * **High-Quality Embeddings**: Leverages the `google/siglip2-base-patch16-naflex` model to generate rich image embeddings.
  * **Efficient Re-indexing**: Save time by only processing new or modified images since the last run. The tool automatically detects changes and updates the index accordingly.
  * **Flexible Configuration**: Use a simple YAML file or command-line arguments to customize everything from input directories and model selection to batch size and hardware acceleration.
  * **Hardware Acceleration**: Supports processing on **CPU** and **CUDA** (NVIDIA GPUs), with an `auto` mode to intelligently select the best available device.
  * **Multiple Output Formats**: Save your index as either a `jsonl` file for easy inspection and parsing or a `pkl` (pickle) file for efficient loading in Python applications.
  * **Resilient and Recoverable**: The indexing process is designed to be robust. If it's interrupted, it creates a `.tmp` file that allows it to resume from where it left off, preventing loss of progress.
  * **Rich Metadata Extraction**: In addition to the vector, the tool extracts and stores useful metadata for each image, including the file path, size, creation/modification times, image dimensions, and EXIF data.

-----

## Installation 💻

To get started, clone the repository and install the required dependencies.

```bash
# Clone the repository (not shown, but assumed)
# pip install -e .
```

The tool requires Python 3.9+ and the dependencies listed in `pyproject.toml`:

  * `torch>=2.0.0`
  * `transformers>=4.30.0`
  * `Pillow>=10.0.0`
  * `PyYAML>=6.0`
  * `tqdm>=4.60.0`
  * `pillow-heif>=0.10.0` (for HEIC/HEIF support)

-----

## Configuration ⚙️

You can configure the Image Indexer using a YAML file. A `config.example.yaml` is provided to get you started.

```yaml
# ----------------------------------------------------
# Example Configuration for the Image Indexer v3.0
# ----------------------------------------------------

# -- Input and Output Paths --
# List of directories to scan for images recursively.
input_dirs:
  - /path/to/your/images/collection1
  # - /path/to/your/images/collection2

# Path to the output index file.
output_file: "image_index.jsonl"

# -- Model and Processing Configuration --
# Hugging Face model name for vision embeddings.
model_name: "google/siglip2-base-patch16-naflex"

# Number of images to process in a single batch.
# Adjust based on your GPU memory.
batch_size: 32

# Hardware device to use.
# Options: "auto", "cpu", "cuda".
# "auto" will attempt to use GPU if available.
device: "auto"

# -- Output Format --
# Format for the output file.
# Options: "jsonl", "pkl"
output_format: "jsonl"

# -- Logging --
# Logging level.
# Options: "DEBUG", "INFO", "WARNING", "ERROR"
log_level: "INFO"
```

**Important Note**: Arguments provided via the command-line interface (CLI) will **override** any values specified in the YAML configuration file.

-----

## Usage 🚀

The tool is run from the command line. You can provide a configuration file or override settings using command-line arguments.

### Basic Indexing

The most straightforward way to run the indexer is by pointing it to your image directories.

```bash
image-indexer -i /path/to/your/images --output-file my_index.jsonl
```

### Using a Configuration File

For more complex setups, use a YAML configuration file.

```bash
image-indexer -c config.yaml
```

-----

## Output Formats and Structure

The tool supports two output formats: `jsonl` and `pkl`.

  * **`jsonl` (JSON Lines)**: This format is human-readable and easy to parse with standard command-line tools. Each line in the file is a valid JSON object representing a single image.
  * **`pkl` (Pickle)**: This is a binary format that serializes Python objects. It is highly efficient for reading and writing in Python, making it ideal for large datasets. The file consists of a sequence of serialized Python dictionary objects, one for each image. To read it, you will need to load the objects from the file in a loop until you reach the end.

Each record saved in the index file, whether in `jsonl` or `pkl` format, is a Python dictionary with the following structure.

```python
{
  "vector": [0.0123, -0.0456, ..., 0.0789],  # List of floats representing the image embedding
  "metadata": {
    "path": "/path/to/your/image.jpg",      # Full path to the image file
    "filename": "image.jpg",               # Name of the image file
    "size_bytes": 123456,                   # Size of the file in bytes
    "creation_time": "2023-10-27T10:00:00", # ISO formatted creation timestamp of the file
    "modification_time": "2023-10-27T10:00:00", # ISO formatted modification timestamp of the file
    "width": 1920,                          # Width of the image in pixels
    "height": 1080,                         # Height of the image in pixels
    "exif": { ... }                         # Dictionary with EXIF data extracted from the image
  }
}
```

-----

## Re-indexing and Recovery

### Re-indexing

To update an existing index with new or modified images, use the `--reindex` flag. The tool will compare the images on disk with the records in the source index and perform the following actions:

  * **New files**: Images found on disk but not in the index will be processed and added.
  * **Modified files**: Images whose modification timestamp on disk is different from the one stored in the index will be re-processed.
  * **Deleted files**: Records in the index that no longer have a corresponding file on disk will be removed.

<!-- end list -->

```bash
image-indexer -c config.yaml --reindex
```

You can also specify a different source index for re-indexing using `--source-index`.

```bash
image-indexer --reindex --source-index /path/to/old_index.jsonl -i /path/to/images -o /path/to/new_index.jsonl
```

### Recovery

If the indexing process is interrupted (e.g., due to a power outage or manual cancellation), the tool will leave a temporary file with a `.tmp` extension. When you run the tool again with the same configuration, it will detect this file and **automatically resume the process** from where it left off, ensuring that no progress is lost.

-----

## Command-Line Arguments

All options in the config file can be overridden via command-line arguments.

| Argument | Description | Default (from config) |
| :--- | :--- | :--- |
| `-c`, `--config-file` | Path to a YAML configuration file. | `None` |
| `-i`, `--input-dirs` | One or more directories to scan for images. | `None` |
| `-o`, `--output-file` | Path for the output index file. | `None` |
| `--model-name` | Hugging Face model name for vision embeddings. | `google/siglip2-base-patch16-naflex` |
| `--batch-size` | Number of images to process in a single batch. | `32` |
| `--device` | Device to use for processing (`auto`, `cpu`, `cuda`). | `auto` |
| `--output-format` | Format for the output file (`jsonl`, `pkl`). | `jsonl` |
| `--log-level` | Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `--reindex` | Enable re-indexing mode. Only processes new or modified files. | `False` |
| `--source-index` | Path to an existing index to use as the source for re-indexing. | `output_file` |

-----

## License 📄

This project is licensed under the MIT License. See the `pyproject.toml` file for details.
