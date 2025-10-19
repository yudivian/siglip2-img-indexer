import argparse
import logging
from . import config
from . import indexer
from . import utils

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(
        description="Image Indexer v3.0 - Create a vector index from image directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config-file",
        type=str,
        help="Path to a YAML configuration file."
    )
    
    parser.add_argument(
        "-i", "--input-dirs",
        action="append", 
        help="One or more directories to scan for images. Can be specified multiple times."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        help="Path for the output index file (e.g., index.jsonl)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Hugging Face model name for vision embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of images to process in a single batch."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for processing."
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jsonl", "pkl"],
        help="Format for the output file."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level."
    )
    
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Enable re-indexing mode. Only processes new or modified files."
    )
    
    parser.add_argument(
        "--source-index",
        type=str,
        help="Path to an existing index to use as the source for re-indexing. Defaults to the output file."
    )

    parser.add_argument(
        "--create-thumbnails",
        action="store_true",
        help="Enable thumbnail creation."
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=str,
        help="Directory to store thumbnails."
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        nargs=2,
        help="Size of thumbnails (width height)."
    )
    
    args = parser.parse_args()
    
    final_config = config.merge_configs(args, parser)
    utils.setup_logging(final_config["log_level"])
    
    logging.debug(f"Final configuration: {final_config}")

    if not final_config.get("input_dirs"):
        logging.error("No input directories specified. Please provide them via --input-dirs or a config file.")
        return

    try:
        idx_runner = indexer.ImageIndexer(final_config)
        idx_runner.run()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()