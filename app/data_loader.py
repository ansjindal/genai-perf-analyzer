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
    engine: str  # tensorrt_llm, vllm, etc.
    gpu_config: str  # h100-fp8, a10g-bf16, etc.
    parallelism: str  # tp1-pp1, tp2, etc.
    optimization: str  # throughput or latency
    concurrency: Optional[int]

    def get_short_name(self) -> str:
        """Get abbreviated name for plot legends."""
        engine_short = 'TRT' if self.engine == 'tensorrt_llm' else self.engine.upper()
        gpu_short = self.gpu_config.split('-')[0].upper()
        precision = self.gpu_config.split('-')[1].upper()
        parallel = self.parallelism.upper()
        return f"{engine_short}-{gpu_short}-{precision}-{parallel}"

class TokenConfig(NamedTuple):
    """Structured representation of token configuration from file name."""
    input_tokens: int
    output_tokens: int
    file_type: str  # 'genai_perf' or 'raw'

    def __hash__(self):
        return hash((self.input_tokens, self.output_tokens, self.file_type))

    def __eq__(self, other):
        if not isinstance(other, TokenConfig):
            return False
        return (self.input_tokens == other.input_tokens and 
                self.output_tokens == other.output_tokens and 
                self.file_type == other.file_type)
    
    def __str__(self):
        return f"Input: {self.input_tokens}, Output: {self.output_tokens}"

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
        
        Format examples:
        1. Top-level: meta_llama-3.1-8b_aws_p5.48xlarge_h100_tensorrt_llm-h100-fp8-tp1-pp1-throughput
        2. Nested: meta_llama-3.1-8b-instruct-openai-chat-concurrency1
        
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
            
            # If this is a nested folder and we have parent info, use info from parent
            if parent_folder:
                parts = parent_folder.split('_')
                # Find cloud provider index
                cloud_idx = -1
                for i, part in enumerate(parts):
                    if part.lower() in ['aws', 'gcp', 'azure']:
                        cloud_idx = i
                        break
                
                if cloud_idx != -1:
                    model = '_'.join(parts[:cloud_idx])
                    cloud = parts[cloud_idx]
                    instance = parts[cloud_idx + 1]
                    gpu = parts[cloud_idx + 2]
                    
                    # Parse model profile from remaining parts
                    profile_parts = '_'.join(parts[cloud_idx + 3:]).split('-')
                    engine = profile_parts[0]
                    gpu_config = '-'.join(profile_parts[1:3])  # h100-fp8 or a10g-bf16
                    parallelism = '-'.join(p for p in profile_parts[3:-1] if p.startswith(('tp', 'pp')))
                    optimization = profile_parts[-1]
                    
                    return ModelConfig(
                        model=folder_name,
                        cloud=cloud,
                        instance=instance,
                        gpu=gpu,
                        engine=engine,
                        gpu_config=gpu_config,
                        parallelism=parallelism,
                        optimization=optimization,
                        concurrency=concurrency
                    )
            
            # For top-level folders, parse directly
            parts = folder_name.split('_')
            cloud_idx = -1
            for i, part in enumerate(parts):
                if part.lower() in ['aws', 'gcp', 'azure']:
                    cloud_idx = i
                    break
            
            if cloud_idx != -1:
                model = '_'.join(parts[:cloud_idx])
                cloud = parts[cloud_idx]
                instance = parts[cloud_idx + 1]
                gpu = parts[cloud_idx + 2]
                
                # Parse model profile from remaining parts
                profile_parts = '_'.join(parts[cloud_idx + 3:]).split('-')
                engine = profile_parts[0]
                gpu_config = '-'.join(profile_parts[1:3])  # h100-fp8 or a10g-bf16
                parallelism = '-'.join(p for p in profile_parts[3:-1] if p.startswith(('tp', 'pp')))
                optimization = profile_parts[-1]
                
                return ModelConfig(
                    model=model,
                    cloud=cloud,
                    instance=instance,
                    gpu=gpu,
                    engine=engine,
                    gpu_config=gpu_config,
                    parallelism=parallelism,
                    optimization=optimization,
                    concurrency=concurrency
                )
            
            # No cloud info found, return with unknown values
            return ModelConfig(
                model=folder_name,
                cloud='unknown',
                instance='unknown',
                gpu='unknown',
                engine='unknown',
                gpu_config='unknown',
                parallelism='unknown',
                optimization='unknown',
                concurrency=concurrency
            )
            
        except Exception as e:
            print(f"Error parsing folder name '{folder_name}': {str(e)}")
            return ModelConfig(
                model=folder_name,
                cloud='unknown',
                instance='unknown',
                gpu='unknown',
                engine='unknown',
                gpu_config='unknown',
                parallelism='unknown',
                optimization='unknown',
                concurrency=concurrency
            )

    def parse_token_config(self, file_name: str) -> Optional[TokenConfig]:
        """Parse token configuration from file name.
        
        Examples: 
        - 200_5_genai_perf.csv -> TokenConfig(200, 5, 'genai_perf')
        - 200_5.json -> TokenConfig(200, 5, 'raw')
        """
        # Try both old and new formats
        patterns = [
            r'(\d+)_(\d+)_genai_perf\.(?:csv|json)',  # new format
            r'(\d+)_(\d+)\.(?:csv|json)'  # old format
        ]
        
        for pattern in patterns:
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

    def get_token_configs(self, run_folder: str) -> set:
        """Get all token configurations from a run folder."""
        configs = set()
        
        # Find the actual path by looking in all top-level directories
        run_path = self.get_run_path(run_folder)
        if not run_path:
            print(f"Warning: Run folder '{run_folder}' not found")
            return configs
        
        # Look for both CSV and JSON files
        for ext in ['csv', 'json']:
            for file in run_path.glob(f'*_genai_perf.{ext}'):
                token_config = self.parse_token_config(file.name)
                if token_config:
                    configs.add(token_config)
        
        if not configs:
            print(f"Warning: No token configurations found in {run_folder}")
        
        return configs

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
            
            # Try file formats in order of preference
            possible_files = [
                (run_path / f"{base_name}_genai_perf.json", "new JSON"),
                (run_path / f"{base_name}_genai_perf.csv", "CSV")
            ]
            
            perf_data = None
            for file_path, format_type in possible_files:
                if file_path.exists():
                    print(f"Found {format_type} file: {file_path}")
                    if file_path.suffix == '.json':
                        perf_data = self.safe_load_json(file_path)
                    else:  # CSV file
                        try:
                            # Read the first section with percentile data
                            df_percentiles = pd.read_csv(file_path, nrows=4)
                            
                            # Read the second section with throughput data
                            df_throughput = pd.read_csv(file_path, skiprows=6)
                            
                            # Convert data to the expected format
                            perf_data = {}
                            
                            # Process percentile metrics
                            metric_name_map = {
                                'Time To First Token (ns)': 'time_to_first_token',
                                'Inter Token Latency (ns)': 'inter_token_latency',
                                'Request Latency (ns)': 'request_latency'
                            }
                            
                            for _, row in df_percentiles.iterrows():
                                if row['Metric'] in metric_name_map:
                                    metric_name = metric_name_map[row['Metric']]
                                    # Convert ns to ms
                                    perf_data[metric_name] = float(row['avg']) / 1_000_000
                            
                            # Process throughput metrics
                            throughput_map = {
                                'Output Token Throughput (per sec)': 'output_token_throughput',
                                'Request Throughput (per sec)': 'request_throughput'
                            }
                            
                            for _, row in df_throughput.iterrows():
                                if row['Metric'] in throughput_map:
                                    metric_name = throughput_map[row['Metric']]
                                    perf_data[metric_name] = float(row['Value'])
                            
                            # Calculate output token throughput per request
                            if 'request_throughput' in perf_data and 'output_token_throughput' in perf_data:
                                perf_data['output_token_throughput_per_request'] = (
                                    perf_data['output_token_throughput'] / perf_data['request_throughput']
                                )
                            
                        except Exception as e:
                            print(f"Error reading CSV file: {str(e)}")
                            continue
                    
                    if perf_data:
                        print(f"Successfully loaded data from {format_type} file")
                        print(f"Processed metrics: {perf_data}")
                        # Store the same data in all three slots for compatibility
                        results[config] = (perf_data, perf_data, perf_data)
                        break
        
        if not results:
            print(f"Warning: No data found for any token configurations in {run_folder}")
        
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

    def get_metrics_for_runs(self, run_folders: List[str], token_config: Optional[TokenConfig] = None) -> Dict[str, Dict[str, Dict]]:
        """Get all metrics for specified runs."""
        metrics = {}
        print(f"\nCalculating metrics for runs: {run_folders}")
        print(f"Looking for metrics: {self.METRICS}")
        print(f"Using token config: {token_config}")
        
        for run_folder in run_folders:
            print(f"\nProcessing run: {run_folder}")
            try:
                # Get the full path to the run folder
                run_path = self.get_run_path(run_folder)
                if not run_path:
                    print(f"Warning: Run folder '{run_folder}' not found")
                    continue
                    
                # Check for file formats
                if token_config:
                    base_name = f"{token_config.input_tokens}_{token_config.output_tokens}"
                    
                    # Try file formats in order of preference
                    possible_files = [
                        (run_path / f"{base_name}_genai_perf.json", "new JSON"),
                        (run_path / f"{base_name}_genai_perf.csv", "CSV")
                    ]
                    
                    perf_data = None
                    for file_path, format_type in possible_files:
                        if file_path.exists():
                            print(f"Found {format_type} file: {file_path}")
                            if file_path.suffix == '.json':
                                perf_data = self.safe_load_json(file_path)
                            else:  # CSV file
                                try:
                                    # Read the first section with percentile data
                                    df_percentiles = pd.read_csv(file_path, nrows=5)
                                    
                                    # Read the second section with throughput data
                                    df_throughput = pd.read_csv(file_path, skiprows=6)
                                    
                                    # Convert data to the expected format
                                    perf_data = {}
                                    
                                    # Process percentile metrics
                                    metric_name_map = {
                                        'Time To First Token (ns)': 'time_to_first_token',
                                        'Inter Token Latency (ns)': 'inter_token_latency',
                                        'Request Latency (ns)': 'request_latency'
                                    }
                                    
                                    for _, row in df_percentiles.iterrows():
                                        if row['Metric'] in metric_name_map:
                                            metric_name = metric_name_map[row['Metric']]
                                            # Store all percentile values
                                            perf_data[metric_name] = {
                                                'avg': float(row['avg']) / 1_000_000,  # Convert ns to ms
                                                'min': float(row['min']) / 1_000_000,
                                                'max': float(row['max']) / 1_000_000,
                                                'p99': float(row['p99']) / 1_000_000,
                                                'p95': float(row['p95']) / 1_000_000,
                                                'p90': float(row['p90']) / 1_000_000,
                                                'p75': float(row['p75']) / 1_000_000,
                                                'p50': float(row['p50']) / 1_000_000,
                                                'p25': float(row['p25']) / 1_000_000,
                                                'unit': 'ms'
                                            }
                                    
                                    # Process throughput metrics
                                    throughput_map = {
                                        'Output Token Throughput (per sec)': 'output_token_throughput',
                                        'Request Throughput (per sec)': 'request_throughput'
                                    }
                                    
                                    for _, row in df_throughput.iterrows():
                                        if row['Metric'] in throughput_map:
                                            metric_name = throughput_map[row['Metric']]
                                            # Store throughput values with unit
                                            perf_data[metric_name] = {
                                                'avg': float(row['Value']),
                                                'min': float(row['Value']),
                                                'max': float(row['Value']),
                                                'p99': float(row['Value']),
                                                'p95': float(row['Value']),
                                                'p90': float(row['Value']),
                                                'p75': float(row['Value']),
                                                'p50': float(row['Value']),
                                                'p25': float(row['Value']),
                                                'unit': 'req/s' if 'Request' in row['Metric'] else 'tokens/sec'
                                            }
                                    
                                    # Calculate output token throughput per request
                                    if 'request_throughput' in perf_data and 'output_token_throughput' in perf_data:
                                        output_per_req = (
                                            perf_data['output_token_throughput']['avg'] / 
                                            perf_data['request_throughput']['avg']
                                        )
                                        perf_data['output_token_throughput_per_request'] = {
                                            'avg': output_per_req,
                                            'min': output_per_req,
                                            'max': output_per_req,
                                            'p99': output_per_req,
                                            'p95': output_per_req,
                                            'p90': output_per_req,
                                            'p75': output_per_req,
                                            'p50': output_per_req,
                                            'p25': output_per_req,
                                            'unit': 'tokens/req'
                                        }
                                    
                                except Exception as e:
                                    print(f"Error reading CSV file: {str(e)}")
                                    import traceback
                                    print(f"Traceback: {traceback.format_exc()}")
                                    continue
                            
                            if perf_data:
                                print(f"Successfully loaded data from {format_type} file")
                                print(f"Processed metrics: {perf_data}")
                                break
                    
                if not perf_data:
                    print(f"Warning: No performance data found for {base_name} in any format")
                    continue
                    
                # Extract metrics
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
                continue
        
        if not metrics:
            print("Warning: No metrics data found for any runs")
        
        print(f"\nFinal metrics: {metrics}")
        return metrics 