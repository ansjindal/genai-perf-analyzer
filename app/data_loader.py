import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
import re

class ModelConfig(NamedTuple):
    """Structured representation of model configuration from directory name."""
    model: str
    cloud: str
    instance: str
    gpu: str
    concurrency: Optional[int]

class TokenConfig(NamedTuple):
    """Structured representation of token configuration from file name."""
    input_tokens: int
    output_tokens: int
    file_type: str  # 'genai_perf' or 'raw'

class GenAIPerfDataLoader:
    """Handles loading and processing of GenAI-Perf data files."""
    
    METRICS = [
        'request_throughput',
        'request_latency',
        'time_to_first_token',
        'inter_token_latency',
        'output_token_throughput',
        'output_token_throughput_per_request'
    ]
    
    METRIC_UNITS = {
        'request_throughput': 'requests/sec',
        'request_latency': 'ms',
        'time_to_first_token': 'ms',
        'inter_token_latency': 'ms',
        'output_token_throughput': 'tokens/sec',
        'output_token_throughput_per_request': 'tokens/sec'
    }
    
    def __init__(self, base_path: str):
        """Initialize the data loader with the base path containing test runs."""
        self.base_path = Path(base_path)
    
    def parse_model_info(self, folder_name: str, parent_folder: str = None) -> ModelConfig:
        """Parse model information from folder name.
        
        Can handle two formats:
        1. Top-level folder: meta_llama-3.1-8b_aws_p5.48xlarge_h100
        2. Nested folder: meta_llama-3.1-8b-instruct-openai-chat-concurrency1
        
        Args:
            folder_name: Name of the folder to parse
            parent_folder: Optional parent folder name for nested directories
        """
        try:
            # First try to extract concurrency if present
            concurrency = None
            if '-concurrency' in folder_name:
                base_parts = folder_name.split('-concurrency')
                folder_name = base_parts[0]
                concurrency = int(base_parts[1])
            
            # If this is a nested folder and we have parent info, use cloud info from parent
            if parent_folder and '_aws_' in parent_folder:
                # Parse parent folder for cloud info
                parts = parent_folder.split('_')
                cloud_idx = parts.index('aws')
                cloud = parts[cloud_idx]
                instance = parts[cloud_idx + 1]
                gpu = parts[cloud_idx + 2]
                
                # For nested folder, everything before -concurrency is model info
                model = folder_name
                
                return ModelConfig(
                    model=model,
                    cloud=cloud,
                    instance=instance,
                    gpu=gpu,
                    concurrency=concurrency
                )
            
            # For top-level folders, parse cloud info directly
            parts = folder_name.split('_')
            cloud_idx = -1
            for i, part in enumerate(parts):
                if part.lower() in ['aws', 'gcp', 'azure']:
                    cloud_idx = i
                    break
            
            if cloud_idx != -1:
                # Top-level folder format
                model = '_'.join(parts[:cloud_idx])
                cloud = parts[cloud_idx]
                instance = parts[cloud_idx + 1]
                gpu = parts[cloud_idx + 2]
                
                return ModelConfig(
                    model=model,
                    cloud=cloud,
                    instance=instance,
                    gpu=gpu,
                    concurrency=concurrency
                )
            else:
                # No cloud info found, return with unknown cloud details
                return ModelConfig(
                    model=folder_name,
                    cloud='unknown',
                    instance='unknown',
                    gpu='unknown',
                    concurrency=concurrency
                )
            
        except Exception as e:
            print(f"Error parsing folder name '{folder_name}': {str(e)}")
            return ModelConfig(
                model=folder_name,
                cloud='unknown',
                instance='unknown',
                gpu='unknown',
                concurrency=concurrency
            )

    def parse_token_config(self, file_name: str) -> Optional[TokenConfig]:
        """Parse token configuration from file name.
        
        Example: 200_5_genai_perf.csv -> TokenConfig(200, 5, 'genai_perf')
        """
        pattern = r'(\d+)_(\d+)(?:_genai_perf)?\.(?:csv|json)'
        match = re.match(pattern, file_name)
        
        if match:
            input_tokens = int(match.group(1))
            output_tokens = int(match.group(2))
            file_type = 'genai_perf' if '_genai_perf' in file_name else 'raw'
            return TokenConfig(input_tokens, output_tokens, file_type)
        return None

    def get_available_test_configs(self) -> List[str]:
        """Return a list of available test configuration folders."""
        # Get all top-level directories
        top_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        # Get all nested test directories
        all_test_dirs = []
        for top_dir in top_dirs:
            nested_dirs = [d.name for d in top_dir.iterdir() if d.is_dir()]
            all_test_dirs.extend(nested_dirs)
        
        return sorted(all_test_dirs, key=lambda x: (
            self.parse_model_info(x, parent_folder=None).model,
            self.parse_model_info(x, parent_folder=None).concurrency or 0
        ))

    def get_token_configs(self, run_folder: str) -> List[TokenConfig]:
        """Get list of available token configurations in a run folder."""
        # First find the actual path by looking in all top-level directories
        run_path = None
        for top_dir in self.base_path.iterdir():
            if top_dir.is_dir():
                test_path = top_dir / run_folder
                if test_path.exists():
                    run_path = test_path
                    break
        
        if not run_path:
            print(f"Warning: Run folder '{run_folder}' not found")
            return []
        
        configs = set()
        for file in run_path.glob('*_genai_perf.csv'):
            token_config = self.parse_token_config(file.name)
            if token_config:
                configs.add(token_config)
        
        return sorted(list(configs), key=lambda x: (x.input_tokens, x.output_tokens))

    def get_parent_folder(self, run_folder: str) -> Optional[str]:
        """Find the parent folder containing the given run folder."""
        for top_dir in self.base_path.iterdir():
            if top_dir.is_dir():
                test_path = top_dir / run_folder
                if test_path.exists():
                    return top_dir.name
        return None

    def get_run_path(self, run_folder: str) -> Optional[Path]:
        """Find the full path to a run folder."""
        for top_dir in self.base_path.iterdir():
            if top_dir.is_dir():
                test_path = top_dir / run_folder
                if test_path.exists():
                    return test_path
        return None

    def group_test_configs(self) -> Dict[str, Dict]:
        """Group test configurations by model and hardware.
        
        Returns:
            Dictionary mapping model name to:
            - model_info: ModelConfig from parent folder
            - test_folders: List of test folder names
        """
        configs = {}
        
        # Get all top-level directories
        top_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        for top_dir in top_dirs:
            # Parse top directory info
            top_info = self.parse_model_info(top_dir.name)
            key = f"{top_info.model}"
            
            if key not in configs:
                configs[key] = {
                    'model_info': top_info,
                    'test_folders': []
                }
            
            # Add all nested test directories
            nested_dirs = [d.name for d in top_dir.iterdir() if d.is_dir()]
            configs[key]['test_folders'].extend(nested_dirs)
            
            # Sort by concurrency
            configs[key]['test_folders'].sort(key=lambda x: 
                self.parse_model_info(x, parent_folder=top_dir.name).concurrency or 0
            )
        
        return configs

    def safe_load_json(self, file_path: Path) -> Dict:
        """Safely load a JSON file with different encodings."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return json.load(f)
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from {file_path} with {encoding} encoding: {str(e)}")
                continue
        print(f"Error: Could not read {file_path} with any supported encoding")
        return {}

    def load_run_data(self, run_folder: str, token_config: Optional[TokenConfig] = None) -> Dict[TokenConfig, Tuple[Dict, Dict, Dict]]:
        """Load data for a specific test run."""
        run_path = self.get_run_path(run_folder)
        if not run_path:
            print(f"Warning: Test run folder '{run_folder}' not found")
            raise FileNotFoundError(f"Test run folder '{run_folder}' not found")
        
        print(f"Loading data from run path: {run_path}")
        
        parent_folder = self.get_parent_folder(run_folder)
        print(f"Parent folder: {parent_folder}")
        
        results = {}
        
        # Load input prompts
        input_data = {}
        input_file = run_path / 'inputs.json'
        if input_file.exists():
            input_data = self.safe_load_json(input_file)
            if input_data:
                print(f"Loaded input prompts from {input_file}")
        else:
            print(f"Warning: Input file not found at {input_file}")
        
        # Get available token configurations
        if token_config:
            configs_to_load = [token_config]
            print(f"Using specified token config: {token_config}")
        else:
            configs_to_load = self.get_token_configs(run_folder)
            print(f"Found token configs: {configs_to_load}")
        
        for config in configs_to_load:
            base_name = f"{config.input_tokens}_{config.output_tokens}"
            print(f"\nProcessing config: {base_name}")
            
            # Load GenAI-Perf results
            genai_perf_file = run_path / f"{base_name}_genai_perf.json"
            genai_perf_data = {}
            if genai_perf_file.exists():
                genai_perf_data = self.safe_load_json(genai_perf_file)
                if genai_perf_data:
                    print(f"Loaded GenAI-Perf data from {genai_perf_file}")
                else:
                    print(f"Warning: Failed to load GenAI-Perf data from {genai_perf_file}")
                    continue
            else:
                print(f"Warning: GenAI-Perf file not found at {genai_perf_file}")
                continue
            
            # Load raw performance data
            raw_file = run_path / f"{base_name}.json"
            raw_data = {}
            if raw_file.exists():
                raw_data = self.safe_load_json(raw_file)
                if raw_data:
                    print(f"Loaded raw data from {raw_file}")
                else:
                    print(f"Warning: Failed to load raw data from {raw_file}")
            else:
                print(f"Warning: Raw data file not found at {raw_file}")
            
            results[config] = (genai_perf_data, genai_perf_data, raw_data)
        
        return results

    def load_multiple_runs(self, run_folders: List[str], token_config: Optional[TokenConfig] = None) -> Dict[str, Dict[TokenConfig, Tuple[Dict, Dict, Dict]]]:
        """Load data for multiple test runs.
        
        Args:
            run_folders: List of run directory names to load
            token_config: Optional specific token configuration to load
            
        Returns:
            Dictionary mapping run names to their data dictionaries
        """
        return {
            run_folder: self.load_run_data(run_folder, token_config)
            for run_folder in run_folders
        }

    def get_metrics_for_runs(self, run_folders: List[str], token_config: Optional[TokenConfig] = None) -> Dict[str, Dict[str, float]]:
        """Get all metrics for specified runs."""
        metrics = {}
        print(f"\nCalculating metrics for runs: {run_folders}")
        print(f"Looking for metrics: {self.METRICS}")
        
        for run_folder in run_folders:
            print(f"\nProcessing run: {run_folder}")
            try:
                run_data = self.load_run_data(run_folder, token_config)
                if run_data:
                    # Use the first token configuration if none specified
                    first_config = token_config or next(iter(run_data.keys()))
                    _, perf_data, _ = run_data[first_config]  # Use the GenAI-Perf JSON data
                    print(f"Available metrics in data: {list(perf_data.keys())}")
                    
                    run_metrics = {}
                    for metric in self.METRICS:
                        if metric in perf_data:
                            metric_data = perf_data[metric]
                            print(f"Found metric {metric}: {metric_data}")
                            run_metrics[metric] = metric_data
                        else:
                            print(f"Warning: Metric '{metric}' not found in performance data")
                            run_metrics[metric] = None
                    
                    metrics[run_folder] = run_metrics
            except Exception as e:
                print(f"Error processing run {run_folder}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
        
        print(f"\nFinal metrics: {metrics}")
        return metrics 