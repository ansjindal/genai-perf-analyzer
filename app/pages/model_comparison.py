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

def main():
    st.title("📊 Model Profile & Configuration Comparison")
    
    # Initialize data loader
    default_data_dir = get_default_data_dir()
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
            # Model selection
            selected_models = st.multiselect(
                "Select Models to Compare",
                options=list(config_groups.keys()),
                format_func=lambda x: f"{config_groups[x]['model_info'].model} ({config_groups[x]['model_info'].cloud.upper()} | {config_groups[x]['model_info'].instance} | {config_groups[x]['model_info'].gpu.upper()})"
            )
            
            if not selected_models:
                st.info("Please select at least one model to analyze.")
                return
                
            # Get common token configs across selected models
            common_token_configs = None
            for model in selected_models:
                model_token_configs = set()
                for run in config_groups[model]['test_folders']:
                    configs = loader.get_token_configs(run)
                    model_token_configs.update(configs)
                
                if common_token_configs is None:
                    common_token_configs = model_token_configs
                else:
                    common_token_configs = common_token_configs.intersection(model_token_configs)
            
            if not common_token_configs:
                st.warning("No common token configurations found across selected models.")
                return
                
            # Token configuration selection
            selected_token_config = st.selectbox(
                "Select Token Configuration",
                options=sorted(list(common_token_configs)),
                format_func=lambda x: f"Input: {x.input_tokens}, Output: {x.output_tokens}"
            )
            
        else:  # Token Configurations comparison
            # Select single model
            selected_model = st.selectbox(
                "Select Model",
                options=list(config_groups.keys()),
                format_func=lambda x: f"{config_groups[x]['model_info'].model} ({config_groups[x]['model_info'].cloud.upper()} | {config_groups[x]['model_info'].instance} | {config_groups[x]['model_info'].gpu.upper()})"
            )
            
            if not selected_model:
                st.info("Please select a model to analyze.")
                return
                
            # Get all token configs for the model
            token_configs = set()
            for run in config_groups[selected_model]['test_folders']:
                configs = loader.get_token_configs(run)
                token_configs.update(configs)
            
            # Token configuration selection
            selected_token_configs = st.multiselect(
                "Select Token Configurations to Compare",
                options=sorted(list(token_configs)),
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
                
                if comparison_type == "Model Profiles":
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
                        
                        # Sort by concurrency
                        model_metrics.sort(key=lambda x: x.get('concurrency', 0))
                        st.session_state.metrics_data[model] = model_metrics
                else:
                    # Load data for all runs
                    for token_config in selected_token_configs:
                        config_metrics = []  # List to store metrics from all runs
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
                        
                        # Sort by concurrency
                        config_metrics.sort(key=lambda x: x.get('concurrency', 0))
                        st.session_state.metrics_data[str(token_config)] = config_metrics
                
                st.session_state.comparison_generated = True
        
        # Only show metrics if comparison was generated
        if st.session_state.get('comparison_generated', False):
            # Create metrics layout
            col1, col2 = st.columns(2)
            
            # Latency Metrics Column
            with col1:
                st.markdown('<h2 class="section-header">Latency Metrics</h2>', unsafe_allow_html=True)
                latency_tabs = st.tabs(['Request Latency', 'Time to First Token', 'Inter Token Latency'])
                
                for tab, metric in zip(latency_tabs, ['request_latency', 'time_to_first_token', 'inter_token_latency']):
                    with tab:
                        # Add statistic parameter selection
                        stat_param = st.selectbox(
                            "Select Statistic Parameter",
                            options=["avg", "p50", "p90", "p95", "p99"],
                            format_func=lambda x: {
                                "avg": "Average",
                                "p50": "Median (p50)",
                                "p90": "90th Percentile",
                                "p95": "95th Percentile",
                                "p99": "99th Percentile"
                            }[x],
                            key=f"{metric}_stat_param"
                        )
                        
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        
                        # Create metric comparison plot
                        if st.session_state.metrics_data:
                            fig = st.session_state.visualizer.create_model_comparison_plot(
                                st.session_state.metrics_data,
                                metric,
                                stat_param=stat_param
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add latency distribution comparison plot
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        
                        if st.session_state.metrics_data:
                            fig = st.session_state.visualizer.create_latency_distribution_comparison_plot(
                                st.session_state.metrics_data,
                                metric
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Throughput Metrics Column
            with col2:
                st.markdown('<h2 class="section-header">Throughput Metrics</h2>', unsafe_allow_html=True)
                throughput_tabs = st.tabs(['Request Throughput', 'Output Token Throughput', 'Output Token Throughput Per Request'])
                
                for tab, metric in zip(throughput_tabs, ['request_throughput', 'output_token_throughput', 'output_token_throughput_per_request']):
                    with tab:
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        
                        # Create plot
                        if st.session_state.metrics_data:
                            fig = st.session_state.visualizer.create_model_comparison_plot(
                                st.session_state.metrics_data,
                                metric,
                                stat_param="avg"  # Always use average for throughput metrics
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error analyzing data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 