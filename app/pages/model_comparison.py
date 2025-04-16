import streamlit as st
import os
from pathlib import Path
from visualizer import GenAIPerfVisualizer
from data_loader import GenAIPerfDataLoader, TokenConfig

st.set_page_config(
    page_title="Model Profile Comparison",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    /* Main container layout */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'SF Pro Display', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2C3E50;
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        border-bottom: 1px solid #E6E9ED;
    }

    /* Plot container */
    .plot-container {
        background-color: white;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #E6E9ED;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def get_default_data_dir():
    """Get the default data directory path."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    return data_dir

def format_model_name(model_info, detailed=False):
    """Format model name with profile information for display."""
    if model_info.engine == 'unknown':
        return f"{model_info.model}"
    
    if detailed:
        # Return detailed configuration for expandable section
        return {
            "Model": model_info.model,
            "Cloud Provider": model_info.cloud.upper(),
            "Instance Type": model_info.instance,
            "GPU": model_info.gpu.upper(),
            "Engine": 'TensorRT-LLM' if model_info.engine == 'tensorrt_llm' else model_info.engine.upper(),
            "GPU Config": model_info.gpu_config.upper(),
            "Parallelism": model_info.parallelism.upper(),
            "Optimization": model_info.optimization.capitalize()
        }
    
    # Create a detailed display string for dropdown
    engine = 'TensorRT-LLM' if model_info.engine == 'tensorrt_llm' else model_info.engine.upper()
    return f"{model_info.model} ({model_info.cloud.upper()} | {model_info.instance} | {model_info.gpu.upper()} | {engine} | {model_info.gpu_config.upper()} | {model_info.parallelism.upper()} | {model_info.optimization.capitalize()})"

def display_model_details(model_info):
    """Display detailed model configuration in an expandable section."""
    details = format_model_name(model_info, detailed=True)
    with st.expander("Model Configuration", expanded=True):  # Set expanded=True to show by default
        # Create a clean layout with columns
        col1, col2 = st.columns(2)
        
        # First column
        with col1:
            st.markdown(f"**Model:** {details['Model']}")
            st.markdown(f"**Cloud Provider:** {details['Cloud Provider']}")
            st.markdown(f"**Instance Type:** {details['Instance Type']}")
            st.markdown(f"**GPU:** {details['GPU']}")
            
        # Second column
        with col2:
            st.markdown(f"**Engine:** {details['Engine']}")
            st.markdown(f"**GPU Config:** {details['GPU Config']}")
            st.markdown(f"**Parallelism:** {details['Parallelism']}")
            st.markdown(f"**Optimization:** {details['Optimization']}")

def main():
    st.title("📊 Model Profile & Configuration Comparison")
    
    # Initialize data loader
    default_data_dir = get_default_data_dir()
    
    # Create a sidebar for data directory input
    with st.sidebar:
        st.header("Data Configuration")
        data_dir = st.text_input(
            "Data Directory Path",
            value=default_data_dir,
            help="Path to directory containing GenAI-Perf test runs"
        )

    if not os.path.exists(data_dir):
        st.warning(f"Data directory not found: {data_dir}")
        return
        
    try:
        loader = GenAIPerfDataLoader(data_dir)
        
        # Get all model configurations
        config_groups = {}
        for top_dir in Path(data_dir).iterdir():
            if top_dir.is_dir():
                model_info = loader.parse_model_info(top_dir.name)
                if model_info:
                    test_folders = []
                    for test_dir in top_dir.iterdir():
                        if test_dir.is_dir():
                            test_folders.append(test_dir.name)
                    if test_folders:
                        config_groups[top_dir.name] = {
                            'model_info': model_info,
                            'test_folders': sorted(test_folders)
                        }
        
        if not config_groups:
            st.warning("No model configurations found.")
            return
            
        # Comparison type selection
        comparison_type = st.radio(
            "Select Comparison Type",
            ["Model Profiles", "Token Configurations"],
            horizontal=True
        )
        
        if comparison_type == "Model Profiles":
            # Add profile filters
            col1, col2, col3, col4 = st.columns(4)
            
            # Get unique values for each filter
            engines = sorted(set(info['model_info'].engine for info in config_groups.values()))
            gpus = sorted(set(info['model_info'].gpu for info in config_groups.values()))
            precisions = sorted(set(info['model_info'].gpu_config.split('-')[1] for info in config_groups.values() if info['model_info'].gpu_config != 'unknown'))
            optimizations = sorted(set(info['model_info'].optimization for info in config_groups.values() if info['model_info'].optimization != 'unknown'))
            
            with col1:
                selected_engines = st.multiselect(
                    "Engine Type",
                    options=engines,
                    format_func=lambda x: 'TensorRT-LLM' if x == 'tensorrt_llm' else x.upper()
                )
            
            with col2:
                selected_gpus = st.multiselect(
                    "GPU Type",
                    options=gpus,
                    format_func=str.upper
                )
            
            with col3:
                selected_precisions = st.multiselect(
                    "Precision",
                    options=precisions,
                    format_func=str.upper
                )
            
            with col4:
                selected_optimizations = st.multiselect(
                    "Optimization",
                    options=optimizations,
                    format_func=str.capitalize
                )
            
            # Filter models based on selections
            filtered_models = list(config_groups.keys())
            if selected_engines:
                filtered_models = [m for m in filtered_models if config_groups[m]['model_info'].engine in selected_engines]
            if selected_gpus:
                filtered_models = [m for m in filtered_models if config_groups[m]['model_info'].gpu in selected_gpus]
            if selected_precisions:
                filtered_models = [m for m in filtered_models if config_groups[m]['model_info'].gpu_config.split('-')[1] in selected_precisions]
            if selected_optimizations:
                filtered_models = [m for m in filtered_models if config_groups[m]['model_info'].optimization in selected_optimizations]
            
            # Model selection
            selected_models = st.multiselect(
                "Select Models to Compare",
                options=filtered_models,
                format_func=lambda x: format_model_name(config_groups[x]['model_info'])
            )
            
            if not selected_models:
                st.info("Please select at least one model to analyze.")
                return
            
            # Display detailed configuration for selected models
            for model in selected_models:
                display_model_details(config_groups[model]['model_info'])
            
            # Get common token configs across selected models
            common_token_configs = None
            for model in selected_models:
                model_token_configs = set()
                for run in config_groups[model]['test_folders']:
                    configs = loader.get_token_configs(run)
                    if configs:  # Only update if we found configs
                        model_token_configs.update(configs)
                
                if model_token_configs:  # Only update common configs if we found any
                    if common_token_configs is None:
                        common_token_configs = model_token_configs
                    else:
                        common_token_configs = common_token_configs.intersection(model_token_configs)
            
            if not common_token_configs:
                st.warning("No common token configurations found across selected models.")
                return
            
            # Convert to sorted list for display
            token_config_list = sorted(list(common_token_configs), key=lambda x: (x.input_tokens, x.output_tokens))
            
            # Token configuration selection
            selected_token_config = st.selectbox(
                "Select Token Configuration",
                options=token_config_list,
                format_func=lambda x: f"Input: {x.input_tokens}, Output: {x.output_tokens}"
            )
            
            # Add analyze button
            if st.button("Generate Comparison"):
                with st.spinner("Loading and analyzing data..."):
                    st.session_state.visualizer = GenAIPerfVisualizer()
                    st.session_state.metrics_data = {}
                    
                    # Load data for each model
                    for model in selected_models:
                        model_metrics = []  # List to store metrics from all runs
                        # Load data from all runs
                        for run in config_groups[model]['test_folders']:
                            run_data = loader.get_metrics_for_runs([run], selected_token_config)
                            if run_data and run in run_data:
                                # Extract concurrency level from run name
                                concurrency = None
                                if '-concurrency' in run:
                                    try:
                                        concurrency = int(run.split('-concurrency')[1])
                                    except ValueError:
                                        continue
                                
                                if concurrency is not None:
                                    # Add concurrency to metrics
                                    metrics = run_data[run]
                                    metrics['concurrency'] = concurrency
                                    model_metrics.append(metrics)
                        
                        if model_metrics:  # Only add if we have data
                            # Sort by concurrency
                            model_metrics.sort(key=lambda x: x.get('concurrency', 0))
                            st.session_state.metrics_data[model] = model_metrics
                    
                    if not st.session_state.metrics_data:
                        st.error("No data found for the selected configuration.")
                        return
                    
                    # Create metrics layout
                    st.markdown('<h2 class="section-header">Performance Comparison</h2>', unsafe_allow_html=True)
                    
                    # Create two columns for side-by-side layout
                    left_col, right_col = st.columns(2)
                    
                    # Latency Metrics Column
                    with left_col:
                        st.markdown('<h3>Latency Metrics</h3>', unsafe_allow_html=True)
                        latency_tabs = st.tabs(['Request Latency', 'Time to First Token', 'Inter Token Latency'])
                        
                        for tab, metric in zip(latency_tabs, ['request_latency', 'time_to_first_token', 'inter_token_latency']):
                            with tab:
                                # Distribution plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_latency_distribution_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Metric comparison plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_model_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Throughput Metrics Column
                    with right_col:
                        st.markdown('<h3>Throughput Metrics</h3>', unsafe_allow_html=True)
                        throughput_tabs = st.tabs(['Request Throughput', 'Output Token Throughput', 'Output Token Throughput Per Request'])
                        
                        for tab, metric in zip(throughput_tabs, ['request_throughput', 'output_token_throughput', 'output_token_throughput_per_request']):
                            with tab:
                                # Distribution plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_latency_distribution_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Metric comparison plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_model_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.comparison_generated = True
                    st.session_state.selected_models = selected_models
                    st.session_state.selected_token_config = selected_token_config
        
        else:
            # Token configuration comparison
            selected_model = st.selectbox(
                "Select Model Configuration",
                options=list(config_groups.keys()),
                format_func=lambda x: format_model_name(config_groups[x]['model_info'])
            )
            
            # Display detailed configuration for selected model
            display_model_details(config_groups[selected_model]['model_info'])
            
            # Get available token configs for the selected model
            token_configs = set()
            for run in config_groups[selected_model]['test_folders']:
                configs = loader.get_token_configs(run)
                token_configs.update(configs)
            
            if not token_configs:
                st.warning("No token configurations found for the selected model.")
                return
            
            # Token configuration selection
            selected_token_configs = st.multiselect(
                "Select Token Configurations to Compare",
                options=sorted(list(token_configs), key=lambda x: (x.input_tokens, x.output_tokens)),
                format_func=lambda x: f"Input: {x.input_tokens}, Output: {x.output_tokens}"
            )
            
            if not selected_token_configs:
                st.info("Please select at least one token configuration to analyze.")
                return
            
            # Add analyze button
            if st.button("Generate Comparison"):
                with st.spinner("Loading and analyzing data..."):
                    st.session_state.visualizer = GenAIPerfVisualizer()
                    st.session_state.metrics_data = {}
                    
                    # Load data for each token configuration
                    for token_config in selected_token_configs:
                        config_metrics = []  # List to store metrics from all runs
                        # Load data from all runs
                        for run in config_groups[selected_model]['test_folders']:
                            run_data = loader.get_metrics_for_runs([run], token_config)
                            if run_data and run in run_data:
                                # Extract concurrency level from run name
                                concurrency = None
                                if '-concurrency' in run:
                                    try:
                                        concurrency = int(run.split('-concurrency')[1])
                                    except ValueError:
                                        continue
                                
                                if concurrency is not None:
                                    # Add concurrency to metrics
                                    metrics = run_data[run]
                                    metrics['concurrency'] = concurrency
                                    config_metrics.append(metrics)
                        
                        if config_metrics:  # Only add if we have data
                            # Sort by concurrency
                            config_metrics.sort(key=lambda x: x.get('concurrency', 0))
                            # Use token config as key
                            config_key = f"Input: {token_config.input_tokens}, Output: {token_config.output_tokens}"
                            st.session_state.metrics_data[config_key] = config_metrics
                    
                    if not st.session_state.metrics_data:
                        st.error("No data found for the selected configurations.")
                        return
                    
                    # Create metrics layout
                    st.markdown('<h2 class="section-header">Performance Comparison</h2>', unsafe_allow_html=True)
                    
                    # Create two columns for side-by-side layout
                    left_col, right_col = st.columns(2)
                    
                    # Latency Metrics Column
                    with left_col:
                        st.markdown('<h3>Latency Metrics</h3>', unsafe_allow_html=True)
                        latency_tabs = st.tabs(['Request Latency', 'Time to First Token', 'Inter Token Latency'])
                        
                        for tab, metric in zip(latency_tabs, ['request_latency', 'time_to_first_token', 'inter_token_latency']):
                            with tab:
                                # Distribution plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_latency_distribution_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Metric comparison plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_model_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Throughput Metrics Column
                    with right_col:
                        st.markdown('<h3>Throughput Metrics</h3>', unsafe_allow_html=True)
                        throughput_tabs = st.tabs(['Request Throughput', 'Output Token Throughput', 'Output Token Throughput Per Request'])
                        
                        for tab, metric in zip(throughput_tabs, ['request_throughput', 'output_token_throughput', 'output_token_throughput_per_request']):
                            with tab:
                                # Distribution plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_latency_distribution_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Metric comparison plot
                                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                                fig = st.session_state.visualizer.create_model_comparison_plot(
                                    st.session_state.metrics_data,
                                    metric
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.comparison_generated = True
                    st.session_state.selected_model = selected_model
                    st.session_state.selected_token_configs = selected_token_configs
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main() 