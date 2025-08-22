"""
Configuration management utilities.
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary or None if loading failed
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None