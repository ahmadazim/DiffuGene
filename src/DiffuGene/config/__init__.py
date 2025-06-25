"""Configuration management for DiffuGene pipeline."""

import os
import yaml
import re
from typing import Dict, Any
from pathlib import Path

def expand_variables(config: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Recursively expand ${variable} references in config values."""
    if context is None:
        context = {}
    
    # Add global config to context for variable expansion
    if 'global' in config:
        context.update(config['global'])
    
    def _expand_value(value):
        if isinstance(value, str):
            # Replace ${section.key} with actual values
            pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_path = match.group(1)
                keys = var_path.split('.')
                
                # Look in context first, then in full config
                result = context
                for key in keys:
                    if isinstance(result, dict) and key in result:
                        result = result[key]
                    else:
                        # Try in full config
                        result = config
                        for k in keys:
                            if isinstance(result, dict) and k in result:
                                result = result[k]
                            else:
                                raise ValueError(f"Variable {var_path} not found in config")
                        break
                return str(result)
            
            return re.sub(pattern, replace_var, value)
        elif isinstance(value, dict):
            return {k: _expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand_value(item) for item in value]
        else:
            return value
    
    return _expand_value(config)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration with variable expansion."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand variables
    config = expand_variables(config)
    
    return config

def get_default_config_path() -> str:
    """Get the default configuration file path."""
    return str(Path(__file__).parent / "pipeline.yaml")

__all__ = ["load_config", "expand_variables", "get_default_config_path"]
