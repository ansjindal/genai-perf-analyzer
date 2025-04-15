import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from data_loader import GenAIPerfDataLoader, TokenConfig

class GenAIPerfVisualizer:
    """Handles creation of visualizations for GenAI-Perf data."""
    
    # Define metric units locally to avoid dependency issues
    METRIC_UNITS = {
        'request_throughput': 'requests/sec',
        'request_latency': 'ms',
        'time_to_first_token': 'ms',
        'inter_token_latency': 'ms',
        'output_token_throughput': 'tokens/sec',
        'output_token_throughput_per_request': 'tokens/sec'
    }
    
    def create_metric_comparison_plot(
        self,
        metrics_data: Dict[str, Dict[str, Dict]],
        metric_name: str,
        sort_by_concurrency: bool = True
    ) -> go.Figure:
        """Create a plot comparing a specific metric across runs, showing available statistics."""
        runs = list(metrics_data.keys())
        
        if sort_by_concurrency:
            # Extract concurrency values and sort
            concurrency_values = []
            for run in runs:
                if 'concurrency' in run:
                    concurrency = int(run.split('concurrency')[1])
                    concurrency_values.append(concurrency)
                else:
                    concurrency_values.append(0)
            
            sorted_indices = np.argsort(concurrency_values)
            runs = [runs[i] for i in sorted_indices]
        
        # Format labels
        run_labels = []
        for run in runs:
            if isinstance(run, TokenConfig):
                # For token configs, show input/output tokens
                run_labels.append(f"In:{run.input_tokens}<br>Out:{run.output_tokens}")
            elif 'TokenConfig' in str(run):
                # Handle string representation of TokenConfig
                parts = str(run).split(',')
                input_tokens = parts[0].split('=')[1]
                output_tokens = parts[1].split('=')[1]
                run_labels.append(f"In:{input_tokens}<br>Out:{output_tokens}")
            else:
                # For model names or other runs, show concurrency if present
                if 'concurrency' in run:
                    concurrency = int(run.split('concurrency')[1])
                    run_labels.append(f"C:{concurrency}")
                else:
                    # Shorten other labels by showing only essential info
                    if '_aws_' in run.lower():
                        # For AWS instances, show GPU type
                        gpu = run.split('_')[-1]
                        run_labels.append(f"GPU:{gpu}")
                    else:
                        run_labels.append(run.split('_')[0])
        
        # Create figure
        fig = go.Figure()
        
        # Get available statistics from first non-empty metric data
        available_stats = []
        metric_unit = self.METRIC_UNITS.get(metric_name, '')  # Default unit from class constant
        
        for run in runs:
            metric_data = metrics_data[run].get(metric_name, {})
            if isinstance(metric_data, dict):
                # Get unit from metric data if available
                if 'unit' in metric_data:
                    metric_unit = metric_data['unit']
                
                # Check which statistics are available
                if 'avg' in metric_data:
                    available_stats.append(('avg', 'Average', 'rgb(158,202,225)'))
                if 'p50' in metric_data:
                    available_stats.append(('p50', 'p50 (median)', 'rgb(94,94,94)'))
                if 'p90' in metric_data:
                    available_stats.append(('p90', 'p90', 'rgb(255,127,14)'))
                if 'p95' in metric_data:
                    available_stats.append(('p95', 'p95', 'rgb(214,39,40)'))
                if 'p99' in metric_data:
                    available_stats.append(('p99', 'p99', 'rgb(148,103,189)'))
                break
        
        print(f"Available stats for {metric_name}: {available_stats}")
        
        # Extract and plot available statistics
        for stat_key, stat_name, stat_color in available_stats:
            values = []
            for run in runs:
                metric_data = metrics_data[run].get(metric_name, {})
                if isinstance(metric_data, dict):
                    value = metric_data.get(stat_key)
                    print(f"Run {run}, Stat {stat_key}: {value}")
                    values.append(value)
                else:
                    values.append(None)
            
            if stat_key == 'avg':
                # Show average as bars
                fig.add_trace(go.Bar(
                    name=stat_name,
                    x=run_labels,
                    y=values,
                    text=[f"{v:.2f}" if v is not None else "N/A" for v in values],
                    textposition='auto',
                    marker_color=stat_color,
                    opacity=0.8
                ))
            else:
                # Show percentiles as lines with markers
                fig.add_trace(go.Scatter(
                    name=stat_name,
                    x=run_labels,
                    y=values,
                    mode='lines+markers',
                    line=dict(color=stat_color, width=2),
                    marker=dict(size=8)
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{metric_name.replace('_', ' ').title()} Comparison Across Runs",
            xaxis_title="Configuration",
            yaxis_title=f"{metric_name.replace('_', ' ').title()} ({metric_unit})",
            template="plotly_white",
            barmode='group',
            # Improve legend
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Configure x-axis for better label display
            xaxis=dict(
                tickangle=0,
                tickmode='array',
                ticktext=run_labels,
                tickvals=list(range(len(run_labels))),
                tickfont=dict(size=10)
            )
        )
        
        return fig
    
    def create_latency_distribution_plot(
        self,
        metrics_data: Dict[str, Dict[str, Dict]],
        metric_name: str
    ) -> go.Figure:
        """Create box plots showing latency distribution across runs.
        
        Args:
            metrics_data: Dictionary containing metrics data for all runs
            metric_name: Name of the latency metric to plot
            
        Returns:
            Plotly figure object
        """
        # Sort runs by concurrency number or token config
        sorted_runs = sorted(
            metrics_data.keys(),
            key=lambda x: int(x.split('concurrency')[1]) if 'concurrency' in x else 0
        )
        
        # Prepare data for plots
        fig = go.Figure()
        
        # Define a custom color palette with better contrast
        colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Yellow-green
            '#17becf'   # Cyan
        ]
        
        for i, run_name in enumerate(sorted_runs):
            metric_data = metrics_data[run_name].get(metric_name, {})
            if not isinstance(metric_data, dict):
                continue
                
            # Get statistics
            stats = {
                'mean': metric_data.get('avg'),
                'median': metric_data.get('p50'),
                'q1': metric_data.get('p25'),
                'q3': metric_data.get('p75'),
                'min': metric_data.get('min'),
                'max': metric_data.get('max'),
                'p90': metric_data.get('p90'),
                'p95': metric_data.get('p95'),
                'p99': metric_data.get('p99'),
                'std': metric_data.get('std')
            }
            
            if not stats['mean']:
                continue
                
            # Generate synthetic points for visualization
            n_points = 100
            if stats['std']:
                points = np.random.normal(stats['mean'], stats['std'], n_points)
                points = np.clip(points, stats['min'], stats['max'])
            else:
                points = np.full(n_points, stats['mean'])
            
            # Format name based on run type
            if isinstance(run_name, str) and 'TokenConfig' in run_name:
                # For token configs, show input/output tokens
                parts = str(run_name).split(',')
                input_tokens = parts[0].split('=')[1]
                output_tokens = parts[1].split('=')[1]
                name = f"In:{input_tokens}, Out:{output_tokens}"
            else:
                # For concurrency levels
                concurrency = int(run_name.split('concurrency')[1]) if 'concurrency' in run_name else 0
                name = f"Concurrency {concurrency}"
            
            color = colors[i % len(colors)]
            
            # Add box plot with points
            fig.add_trace(go.Box(
                y=points,
                name=name,
                boxpoints='outliers',  # show outliers only
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.7
                ),
                line=dict(
                    color=color,
                    width=2
                ),
                fillcolor=color,
                opacity=0.6,
                # Add detailed statistics to hover
                customdata=np.array([[
                    stats['mean'],
                    stats['median'],
                    stats['q1'],
                    stats['q3'],
                    stats['p90'],
                    stats['p95'],
                    stats['p99']
                ]]),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Mean: %{customdata[0]:.2f}<br>" +
                    "Median: %{customdata[1]:.2f}<br>" +
                    "Q1: %{customdata[2]:.2f}<br>" +
                    "Q3: %{customdata[3]:.2f}<br>" +
                    "P90: %{customdata[4]:.2f}<br>" +
                    "P95: %{customdata[5]:.2f}<br>" +
                    "P99: %{customdata[6]:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Get metric unit
        unit = metric_data.get('unit', self.METRIC_UNITS.get(metric_name, ''))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{metric_name.replace('_', ' ').title()}",  # Simple title without bold
                x=0.5,
                y=0.95,  # Move down slightly
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=14,  # Smaller font size
                    color='rgb(50, 50, 50)',  # Dark gray color
                    family="Arial, sans-serif"
                )
            ),
            xaxis_title="Configuration",
            yaxis_title=f"{metric_name.replace('_', ' ').title()} ({unit})",
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title=None,
                font=dict(size=10)
            ),
            # Add hover mode
            hovermode='closest',
            # Add grid
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=0,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            # Add margin
            margin=dict(t=40, b=50, l=50, r=50),  # Reduced top margin
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_throughput_over_concurrency_plot(
        self,
        metrics_data: Dict[str, Dict[str, Dict]],
        metric_name: str
    ) -> go.Figure:
        """Create a line plot showing throughput vs concurrency."""
        concurrency_values = []
        throughput_values = []
        
        # Sort runs by concurrency number
        sorted_runs = sorted(
            metrics_data.keys(),
            key=lambda x: int(x.split('concurrency')[1]) if 'concurrency' in x else 0
        )
        
        print(f"\nProcessing throughput data:")
        print(f"Available runs: {sorted_runs}")
        
        for run_name in sorted_runs:
            try:
                # Extract concurrency number
                if 'concurrency' in run_name:
                    concurrency = int(run_name.split('concurrency')[1])
                    
                    # Get metric data
                    metric_data = metrics_data[run_name].get(metric_name, {})
                    print(f"Run {run_name}, Metric data: {metric_data}")
                    
                    if isinstance(metric_data, dict) and 'avg' in metric_data:
                        throughput = metric_data['avg']
                        print(f"Concurrency {concurrency}: {throughput}")
                        concurrency_values.append(concurrency)
                        throughput_values.append(throughput)
            except (ValueError, TypeError) as e:
                print(f"Error processing run {run_name}: {str(e)}")
                continue
        
        if not concurrency_values or not throughput_values:
            print("No valid throughput data found")
            return None
        
        print(f"Final data points:")
        print(f"Concurrency values: {concurrency_values}")
        print(f"Throughput values: {throughput_values}")
        
        # Create the plot
        fig = go.Figure(data=[
            go.Scatter(
                x=concurrency_values,
                y=throughput_values,
                mode='lines+markers',
                name=metric_name,
                line=dict(color='rgb(31, 119, 180)', width=2),
                marker=dict(size=10)
            )
        ])
        
        # Get metric unit
        unit = metrics_data[sorted_runs[0]][metric_name].get('unit', self.METRIC_UNITS.get(metric_name, ''))
        
        # Update layout
        fig.update_layout(
            title=f"{metric_name.replace('_', ' ').title()} vs Concurrency",
            xaxis_title="Concurrency",
            yaxis_title=f"{metric_name.replace('_', ' ').title()} ({unit})",
            template="plotly_white",
            showlegend=False,
            # Add hover mode
            hovermode='x unified',
            # Add grid
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                tickmode='linear'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='LightGray'
            )
        )
        
        return fig
    
    def create_metric_timeline_plot(
        self,
        run_data_dict: Dict[str, Dict],
        metric_name: str
    ) -> go.Figure:
        """Create a line plot showing metric values over time for all concurrency levels."""
        fig = go.Figure()
        
        # Sort runs by concurrency number
        sorted_runs = sorted(
            run_data_dict.keys(),
            key=lambda x: int(x.split('concurrency')[1]) if 'concurrency' in x else 0
        )
        
        # Color scale for different concurrency levels
        colors = px.colors.qualitative.Set2
        
        for i, run_name in enumerate(sorted_runs):
            metric_data = run_data_dict[run_name].get(metric_name, {})
            if not isinstance(metric_data, dict) or 'avg' not in metric_data:
                continue
            
            # Get statistics
            mean = metric_data['avg']
            std = metric_data.get('std', 0)
            min_val = metric_data.get('min', mean - 2*std)
            max_val = metric_data.get('max', mean + 2*std)
            
            # Generate synthetic timeline data
            n_points = 100
            x = np.arange(n_points)
            y = np.random.normal(mean, std/4, n_points)
            y = np.clip(y, min_val, max_val)
            
            # Extract concurrency number for legend
            concurrency = int(run_name.split('concurrency')[1]) if 'concurrency' in run_name else 0
            
            # Add trace for this concurrency level
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'{concurrency}',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                )
            ))
        
        # Get metric unit
        unit = metric_data.get('unit', self.METRIC_UNITS.get(metric_name, ''))
        
        # Update layout
        fig.update_layout(
            title=f"{metric_name.replace('_', ' ').title()} Timeline Across Concurrency Levels",
            xaxis_title="Sample Index",
            yaxis_title=f"{metric_name.replace('_', ' ').title()} ({unit})",
            template="plotly_white",
            # Improve legend
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add hover mode
            hovermode='x unified'
        )
        
        return fig 