import os
import json
import streamlit as st
from pathlib import Path
from visualizer import GenAIPerfVisualizer
from data_loader import GenAIPerfDataLoader, TokenConfig

st.set_page_config(
    page_title="GenAI Performance Analyzer",
    page_icon="📊",
    layout="wide"
)

def get_default_data_dir():
    """Get the default data directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    return data_dir

def format_model_info(info) -> str:
    """Format model information for display."""
    if not info:
        return "Model information not available"
    
    return f"""
    - **Model**: {info.model or 'Unknown'}
    - **Cloud**: {(info.cloud or 'Unknown').upper()}
    - **Instance**: {info.instance or 'Unknown'}
    - **GPU**: {(info.gpu or 'Unknown').upper()}
    """

def format_token_config(config: TokenConfig) -> str:
    """Format token configuration for display."""
    if not config:
        return "Token configuration not available"
    return f"Input: {config.input_tokens}, Output: {config.output_tokens}"

def format_concurrency(model_info) -> str:
    """Format concurrency information for display."""
    if not model_info or model_info.concurrency is None:
        return "Concurrency: Unknown"
    return f"Concurrency: {model_info.concurrency}"

def display_model_details(model_info):
    """Display detailed model configuration in the sidebar."""
    st.sidebar.markdown("### Model Configuration")
    
    # Create two columns for compact display
    col1, col2 = st.sidebar.columns(2)
    
    # First column - Hardware details
    col1.markdown(f"**Cloud:** {model_info.cloud.upper()}")
    col1.markdown(f"**Instance:** {model_info.instance}")
    col1.markdown(f"**GPU:** {model_info.gpu.upper()}")
    
    # Second column - Software/Config details
    engine_name = 'TensorRT-LLM' if model_info.engine == 'tensorrt_llm' else model_info.engine.upper()
    col2.markdown(f"**Engine:** {engine_name}")
    col2.markdown(f"**Config:** {model_info.gpu_config.upper()}")
    col2.markdown(f"**Parallel:** {model_info.parallelism}")

    # Optimization as a single line below the columns
    st.sidebar.markdown(f"**Optimization:** {model_info.optimization.capitalize()}")

def format_model_name(model_info, detailed=False):
    """Format model name with profile information for display."""
    if model_info.engine == 'unknown':
        return f"{model_info.model}"
    
    # Create a display string for dropdown with full model name
    engine = 'TRT' if model_info.engine == 'tensorrt_llm' else model_info.engine.upper()
    return f"{model_info.model} ({model_info.cloud.upper()} | {model_info.instance} | {model_info.gpu} | {engine} | {model_info.gpu_config.upper()} | {model_info.parallelism})"

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        /* Main container layout */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }

        /* Control panel (sidebar) styling */
        [data-testid="stSidebar"] {
            background-color: #F5F7FA;
            border-right: 1px solid #E6E9ED;
        }

        /* Main content area */
        .main-content {
            display: flex;
            flex-direction: row;
            gap: 1rem;
            padding: 1rem;
        }

        /* Column styling */
        .column {
            flex: 1;
            min-width: 0;  /* Allow columns to shrink below content size */
            padding: 0 0.5rem;
        }

        /* Main title styling */
        .main-title {
            font-family: 'SF Pro Display', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E88E5;
            padding: 1rem;
            margin: 0;
            border-bottom: 2px solid #E6E9ED;
            background-color: #FFFFFF;
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

        /* Ensure plots take full width of their container */
        .js-plotly-plot {
            width: 100% !important;
        }

        /* Input fields and selectors */
        .stTextInput > div[data-baseweb="input"] {
            background-color: white !important;
            border-color: #E6E9ED !important;
        }
        
        .stSelectbox > div[data-baseweb="select"] {
            background-color: white !important;
            border-color: #E6E9ED !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 1rem;
            font-weight: 500;
            color: #1976D2;
            background-color: #FFFFFF;
            border: 1px solid #E6E9ED;
            border-radius: 4px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        
        /* Model info styling */
        .model-info {
            font-family: 'SF Pro Text', sans-serif;
            font-size: 1rem;
            line-height: 1.6;
            color: #424242;
            background-color: #FFFFFF;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #E6E9ED;
            margin: 0.5rem 0;
            width: 100%;
        }

        /* Form elements spacing */
        .element-container {
            margin: 0.5rem 0 !important;
        }

        /* Adjust spacing for selectbox */
        .stSelectbox > div > div {
            padding: 0.25rem !important;
        }

        /* Remove padding from expander content */
        .streamlit-expanderContent {
            padding: 0.5rem 0 !important;
        }

        /* Warning and info messages */
        .stAlert {
            background-color: white !important;
            border: 1px solid #E6E9ED !important;
            margin: 0.5rem 0 !important;
        }

        /* Help text */
        .stMarkdown div.help-text {
            font-size: 0.9rem;
            color: #64748B;
            margin: 0.25rem 0;
        }

        /* Metrics grid layout */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            padding: 0.5rem;
        }

        /* Metrics column styling */
        .metrics-column {
            background-color: #FFFFFF;
            padding: 0.5rem;
        }

        /* Resize handle styling */
        [data-testid="stHorizontalBlock"] > div:first-child::-webkit-resizer {
            background-color: #E6E9ED;
            border: 1px solid #D1D5DB;
            border-radius: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title
    st.markdown('<h1 class="main-title">📊 GenAI Performance Analyzer</h1>', unsafe_allow_html=True)
    
    # Use sidebar for control panel
    with st.sidebar:
        st.markdown('<h2 class="section-header">Data Selection</h2>', unsafe_allow_html=True)
        
        # Data directory input with default path
        default_data_dir = get_default_data_dir()
        data_dir = st.text_input(
            "Data Directory Path",
            value=default_data_dir,
            help="Path to directory containing GenAI-Perf test runs"
        )
        
        # Show instructions when no data is found
        if not os.path.exists(data_dir):
            st.warning(
                f"Data directory not found: {data_dir}\n\n"
                "Please create the directory and add your GenAI-Perf test run folders to it."
            )
            st.markdown("""
            ### How to Add Data
            1. Create the data directory if it doesn't exist
            2. Create a model directory with hardware info:
               `model_cloud_instance_gpu`
               Example: `meta_llama-3.1-8b_aws_p5.48xlarge_h100`
            3. Inside the model directory, add concurrency test folders:
               `model-variant-api_type-concurrencyN`
               Example: `meta_llama-3.1-8b-instruct-openai-chat-concurrency1`
            4. Each concurrency folder should contain:
               - Performance data files with naming convention:
                 `{input_tokens}_{output_tokens}_genai_perf.csv`
                 `{input_tokens}_{output_tokens}_genai_perf.json`
                 `{input_tokens}_{output_tokens}.json`
               - Input prompts file: `inputs.json`
            """)
            return
        
        try:
            # Initialize data loader
            loader = GenAIPerfDataLoader(data_dir)
            
            # Group test configurations
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
                st.warning(
                    "No test configurations found in the data directory. Please add your test data to:\n\n"
                    f"`{data_dir}`"
                )
                return
            
            # Model configuration selection
            selected_config = st.selectbox(
                "Select Model Configuration",
                options=list(config_groups.keys()),
                format_func=lambda x: format_model_name(config_groups[x]['model_info'], detailed=True)
            )
            
            # Get model info and available runs
            model_info = config_groups[selected_config]['model_info']
            available_runs = config_groups[selected_config]['test_folders']
            
            if not available_runs:
                st.warning(
                    f"No concurrency tests found for {selected_config}. Please add test folders to:\n\n"
                    f"`{data_dir}/{selected_config}`"
                )
                return
            
            # Display model configuration details in sidebar
            display_model_details(model_info)
            
            # Token configuration section
            st.markdown('<h2 class="section-header">Token Configuration</h2>', unsafe_allow_html=True)
            
            # Get available token configurations from first run
            first_run = available_runs[0]
            token_configs = loader.get_token_configs(first_run)
            
            if not token_configs:
                st.warning(
                    f"No performance data files found in the test folders. Each folder should contain files like:\n\n"
                    "- `200_5_genai_perf.csv`\n"
                    "- `200_5_genai_perf.json`\n"
                    "- `200_5.json`"
                )
                return
            
            # Token configuration selection
            selected_token_config = st.selectbox(
                "Select Token Configuration",
                options=token_configs,
                format_func=format_token_config,
                help="Select the input/output token combination to analyze"
            )
            
            # Run selection - automatically select all runs
            selected_runs = available_runs
            
            # Add an Analyze button
            if st.button("Analyze Selected Configuration"):
                with st.spinner("Loading and analyzing data..."):
                    # Load data for selected runs
                    run_data = loader.load_multiple_runs(selected_runs, selected_token_config)
                    metrics_data = loader.get_metrics_for_runs(selected_runs, selected_token_config)
                    
                    # Store the loaded data and configuration in session state
                    st.session_state.run_data = run_data
                    st.session_state.metrics_data = metrics_data
                    st.session_state.selected_runs = selected_runs
                    st.session_state.selected_token_config = selected_token_config
                    st.session_state.selected_config = selected_config
                    st.session_state.data_dir = data_dir
                    
                    # Show success message
                    st.success("Data loaded and analyzed successfully!")
            
            # Only show additional data section if analysis has been run
            if 'metrics_data' in st.session_state and selected_config == st.session_state.selected_config:
                st.markdown('<h2 class="section-header">Additional Data</h2>', unsafe_allow_html=True)
                data_type = st.selectbox(
                    "Select Data to View",
                    options=[
                        "Run Configurations",
                        "Raw Performance Data",
                        "Input Prompts"
                    ]
                )
                
                if data_type == "Run Configurations":
                    for run_name in st.session_state.selected_runs:
                        model_info = loader.parse_model_info(run_name, parent_folder=selected_config)
                        with st.expander(f"Configuration: {format_concurrency(model_info)}"):
                            run_data_dict = st.session_state.run_data[run_name]
                            _, config, _ = run_data_dict[st.session_state.selected_token_config]
                            st.json(config)
                elif data_type == "Raw Performance Data":
                    for run_name in st.session_state.selected_runs:
                        model_info = loader.parse_model_info(run_name, parent_folder=selected_config)
                        with st.expander(f"Raw Data: {format_concurrency(model_info)}"):
                            run_data_dict = st.session_state.run_data[run_name]
                            _, _, raw_data = run_data_dict[st.session_state.selected_token_config]
                            st.json(raw_data)
                elif data_type == "Input Prompts":
                    for run_name in st.session_state.selected_runs:
                        model_info = loader.parse_model_info(run_name, parent_folder=selected_config)
                        with st.expander(f"Inputs: {format_concurrency(model_info)}"):
                            input_file = Path(data_dir) / run_name / 'inputs.json'
                            if input_file.exists():
                                with open(input_file, 'r') as f:
                                    inputs = json.load(f)
                                    st.json(inputs)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return
    
    # Main content area with metrics
    # Only show metrics if data has been loaded AND the selected configuration matches
    if ('metrics_data' in st.session_state and 
        'selected_config' in st.session_state and 
        selected_config == st.session_state.selected_config):
        
        # Initialize visualizer
        visualizer = GenAIPerfVisualizer()
        
        # Create two columns for the metrics
        col1, col2 = st.columns(2)
        
        # Latency Metrics Column
        with col1:
            st.markdown('<h2 class="section-header">Latency Metrics</h2>', unsafe_allow_html=True)
            metric_tabs = st.tabs(['Request Latency', 'Time to First Token', 'Inter Token Latency'])
            
            for tab, metric in zip(metric_tabs, ['request_latency', 'time_to_first_token', 'inter_token_latency']):
                with tab:
                    plot_type = st.selectbox(
                        "Select Plot Type",
                        options=["Metric Comparison", "Latency Distribution", "Metric Timeline"],
                        key=f"{metric}_plot_type"
                    )
                    
                    # Only create plot when plot type is selected
                    if plot_type:
                        st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                        if plot_type == "Metric Comparison":
                            fig = visualizer.create_metric_comparison_plot(st.session_state.metrics_data, metric)
                        elif plot_type == "Latency Distribution":
                            fig = visualizer.create_latency_distribution_plot(st.session_state.metrics_data, metric)
                        else:  # Metric Timeline
                            fig = visualizer.create_metric_timeline_plot(st.session_state.metrics_data, metric)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Throughput Metrics Column
        with col2:
            st.markdown('<h2 class="section-header">Throughput Metrics</h2>', unsafe_allow_html=True)
            metric_tabs = st.tabs(['Request Throughput', 'Output Token Throughput', 'Output Token Throughput Per Request'])
            
            for tab, metric in zip(metric_tabs, ['request_throughput', 'output_token_throughput', 'output_token_throughput_per_request']):
                with tab:
                    plot_type = st.selectbox(
                        "Select Plot Type",
                        options=["Metric Comparison", "Throughput vs Concurrency", "Metric Timeline"],
                        key=f"{metric}_plot_type"
                    )
                    
                    # Only create plot when plot type is selected
                    if plot_type:
                        st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                        if plot_type == "Metric Comparison":
                            fig = visualizer.create_metric_comparison_plot(st.session_state.metrics_data, metric)
                        elif plot_type == "Throughput vs Concurrency":
                            fig = visualizer.create_throughput_over_concurrency_plot(st.session_state.metrics_data, metric)
                        else:  # Metric Timeline
                            fig = visualizer.create_metric_timeline_plot(st.session_state.metrics_data, metric)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 