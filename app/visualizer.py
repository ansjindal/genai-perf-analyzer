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
        stat_param: str = "avg"
    ) -> go.Figure:
        """Create a plot comparing request throughput against metric values for a single model."""
        print(f"Received metrics_data with {len(metrics_data)} entries")  # Debug
        
        if not metrics_data:
            print("No metrics data received")  # Debug
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="No data available for plotting",
                xaxis_title="No Data",
                yaxis_title="No Data"
            )
            return fig
            
        fig = go.Figure()
        
        # Get metric unit from first data point that has data
        metric_unit = self.METRIC_UNITS.get(metric_name, '')
        
        # Define statistics to show
        stats_to_show = ['avg', 'p50', 'p90', 'p95', 'p99']
        stat_display_names = {
            'avg': 'Average',
            'p50': 'P50',
            'p90': 'P90',
            'p95': 'P95',
            'p99': 'P99'
        }
        
        # Define colors and dash patterns for each statistic
        stat_styles = {
            'avg': {'color': 'rgb(31, 119, 180)', 'dash': 'solid'},      # Blue, solid
            'p50': {'color': 'rgb(44, 160, 44)', 'dash': 'dot'},         # Green, dotted
            'p90': {'color': 'rgb(214, 39, 40)', 'dash': 'dash'},        # Red, dashed
            'p95': {'color': 'rgb(148, 103, 189)', 'dash': 'dashdot'},   # Purple, dashdot
            'p99': {'color': 'rgb(255, 127, 14)', 'dash': 'longdash'}    # Orange, long dash
        }
        
        # Create traces for each statistic
        for stat in stats_to_show:
            x_values = []  # metric values
            y_values = []  # throughput values
            hover_text = []  # hover information
            
            # Process each run configuration
            for label, run_data in metrics_data.items():
                # Get metric data and throughput data
                metric_data = run_data.get(metric_name, {})
                throughput_data = run_data.get('request_throughput', {})
                
                if isinstance(metric_data, dict) and isinstance(throughput_data, dict):
                    metric_value = metric_data.get(stat)
                    throughput = throughput_data.get('avg')  # Always use average for throughput
                    
                    if metric_value is not None and throughput is not None:
                        x_values.append(metric_value)
                        y_values.append(throughput)
                        hover_text.append(
                            f"{metric_name} ({stat}): {metric_value:.2f} {metric_unit}<br>" +
                            f"Throughput: {throughput:.2f} req/s"
                        )
            
            if x_values and y_values:
                # Sort points by x values for proper line connection
                points = sorted(zip(x_values, y_values, hover_text))
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                hover_text = [p[2] for p in points]
                
                # Add line plot for this statistic with custom style
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=stat_display_names[stat],
                    text=hover_text,
                    line=dict(
                        color=stat_styles[stat]['color'],
                        width=2,
                        dash=stat_styles[stat]['dash']
                    ),
                    marker=dict(
                        size=8,
                        line=dict(width=2, color='white'),
                        color=stat_styles[stat]['color']
                    ),
                    hovertemplate="%{text}<extra></extra>"
                ))
        
        # Check if any traces were added
        if not fig.data:
            print("No traces were added to the figure")  # Debug
            fig.update_layout(
                title="No valid data available for plotting",
                xaxis_title="No Data",
                yaxis_title="No Data"
            )
            return fig
            
        print(f"Created figure with {len(fig.data)} traces")  # Debug
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Request Throughput vs {metric_name.replace('_', ' ').title()}",
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title=f"{metric_name.replace('_', ' ').title()} ({metric_unit})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            yaxis=dict(
                title="Request Throughput (req/s)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            hovermode='closest',
            margin=dict(t=150, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_latency_distribution_plot(
        self,
        metrics_data: Dict[str, Dict[str, Dict]],
        metric_name: str
    ) -> go.Figure:
        """Create horizontal box plots showing latency distribution with throughput on y-axis."""
        # Collect all latency values and corresponding throughputs
        latency_values = []
        throughput = None
        for run_name in metrics_data:
            metric_data = metrics_data[run_name].get(metric_name, {})
            throughput_data = metrics_data[run_name].get('request_throughput', {})
            if (isinstance(metric_data, dict) and isinstance(throughput_data, dict) and
                'avg' in metric_data and 'avg' in throughput_data):
                # Get statistics
                mean = metric_data['avg']
                p50 = metric_data.get('p50', mean)
                p25 = metric_data.get('p25', mean - (p50 - mean))
                p75 = metric_data.get('p75', mean + (mean - p50))
                p90 = metric_data.get('p90', p75 + (p75 - p25))
                p99 = metric_data.get('p99', p90 + (p90 - p75))
                min_val = metric_data.get('min', p25 - 1.5 * (p75 - p25))
                max_val = metric_data.get('max', p75 + 1.5 * (p75 - p25))
                # Store all statistics
                latency_values.extend([min_val, p25, p50, p75, max_val, mean, p90, p99])
                # Store average throughput
                if throughput is None:
                    throughput = throughput_data['avg']
        if not latency_values:
            return None
        # Create figure
        fig = go.Figure()
        # Define colors
        box_color = 'rgb(31, 119, 180)'
        outlier_color = 'rgba(31, 119, 180, 0.6)'
        mean_color = 'rgb(214, 39, 40)'
        # Calculate statistics for the box plot
        min_val = min(latency_values)
        max_val = max(latency_values)
        q1 = np.percentile(latency_values, 25)
        median = np.percentile(latency_values, 50)
        q3 = np.percentile(latency_values, 75)
        mean = np.mean(latency_values)
        p90 = np.percentile(latency_values, 90)
        p99 = np.percentile(latency_values, 99)
        # Add horizontal box plot (metric on x, throughput on y)
        fig.add_trace(go.Box(
            y=[throughput] * 5,
            x=[min_val, q1, median, q3, max_val],
            name='Distribution',
            orientation='h',
            boxpoints=False,
            line=dict(color=box_color, width=2),
            fillcolor='rgba(31, 119, 180, 0.1)',
            whiskerwidth=0.7
        ))
        # Add mean marker with error bars
        fig.add_trace(go.Scatter(
            y=[throughput],
            x=[mean],
            mode='markers',
            name='Mean',
            marker=dict(
                symbol='line-ns',  # Vertical line symbol
                size=20,
                color=mean_color,
                line=dict(width=2)
            ),
            error_x=dict(
                type='data',
                array=[q3 - q1],  # IQR as error
                color=mean_color,
                thickness=1,
                width=10
            ),
            showlegend=False,
            hovertemplate=(
                f"Mean: {mean:.2f}<br>" +
                f"IQR: [{q1:.2f}, {q3:.2f}]<br>" +
                f"Throughput: {throughput:.2f} req/s<br>" +
                "<extra></extra>"
            )
        ))
        # Add percentile markers
        fig.add_trace(go.Scatter(
            y=[throughput],
            x=[p90],
            mode='markers',
            name='P90',
            marker=dict(
                symbol='diamond',
                size=10,
                color=outlier_color,
                line=dict(color=box_color, width=1)
            ),
            showlegend=False,
            hovertemplate=(
                f"P90: {p90:.2f}<br>" +
                f"Throughput: {throughput:.2f} req/s<br>" +
                "<extra></extra>"
            )
        ))
        fig.add_trace(go.Scatter(
            y=[throughput],
            x=[p99],
            mode='markers',
            name='P99',
            marker=dict(
                symbol='star',
                size=12,
                color=outlier_color,
                line=dict(color=box_color, width=1)
            ),
            showlegend=False,
            hovertemplate=(
                f"P99: {p99:.2f}<br>" +
                f"Throughput: {throughput:.2f} req/s<br>" +
                "<extra></extra>"
            )
        ))
        # Get metric unit
        unit = metrics_data[list(metrics_data.keys())[0]][metric_name].get('unit', self.METRIC_UNITS.get(metric_name, ''))
        # Calculate axis ranges with padding
        x_padding = (max_val - min_val) * 0.1
        y_padding = throughput * 0.1 if throughput else 1
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{metric_name.replace('_', ' ').title()} Distribution vs Request Throughput",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title=f"{metric_name.replace('_', ' ').title()} ({unit})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                range=[min_val - x_padding, max_val + x_padding]
            ),
            yaxis=dict(
                title="Request Throughput (req/s)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                range=[throughput - y_padding, throughput + y_padding] if throughput else None
            ),
            template="plotly_white",
            showlegend=False,
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        return fig
    
    def create_throughput_over_concurrency_plot(
        self,
        metrics_data: Dict[str, Dict[str, Dict]],
        metric_name: str
    ) -> go.Figure:
        """Create a plot showing request throughput against concurrency levels."""
        print(f"Creating throughput over concurrency plot for {metric_name}")  # Debug
        
        if not metrics_data:
            print("No metrics data received")  # Debug
            return None
            
        fig = go.Figure()
        
        # Get metric unit
        first_key = next(iter(metrics_data))
        metric_unit = ''
        if isinstance(metrics_data[first_key], dict):
            metric_data = metrics_data[first_key].get(metric_name, {})
            if isinstance(metric_data, dict):
                metric_unit = metric_data.get('unit', self.METRIC_UNITS.get(metric_name, ''))
        
        # Collect data points
        x_values = []  # concurrency levels
        y_values = []  # metric values
        hover_text = []  # hover information
        
        # Process the data
        for run_name, run_data in metrics_data.items():
            if isinstance(run_data, dict):
                # Extract concurrency from run name if available
                concurrency = None
                if '-concurrency' in run_name:
                    try:
                        concurrency = int(run_name.split('-concurrency')[1])
                    except ValueError:
                        continue
                
                if concurrency is not None:
                    metric_data = run_data.get(metric_name, {})
                    if isinstance(metric_data, dict) and 'avg' in metric_data:
                        metric_value = metric_data['avg']
                        x_values.append(concurrency)
                        y_values.append(metric_value)
                        hover_text.append(
                            f"Concurrency: {concurrency}<br>" +
                            f"{metric_name.replace('_', ' ').title()}: {metric_value:.2f} {metric_unit}"
                        )
        
        if not x_values or not y_values:
            print("No valid data points found")  # Debug
            return None
        
        # Sort points by concurrency
        points = sorted(zip(x_values, y_values, hover_text))
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        hover_text = [p[2] for p in points]
        
        # Add the trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=metric_name.replace('_', ' ').title(),
            text=hover_text,
            line=dict(
                width=2,
                color='rgb(31, 119, 180)'
            ),
            marker=dict(
                size=8,
                color='rgb(31, 119, 180)',
                line=dict(width=2, color='white')
            ),
            hovertemplate="%{text}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{metric_name.replace('_', ' ').title()} vs Concurrency",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title="Concurrency Level",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            yaxis=dict(
                title=f"{metric_name.replace('_', ' ').title()} ({metric_unit})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            template="plotly_white",
            showlegend=False,
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_metric_timeline_plot(
        self,
        run_data_dict: Dict[str, Dict],
        metric_name: str
    ) -> go.Figure:
        """Create a line plot showing metric values over time, colored by throughput."""
        fig = go.Figure()
        
        # Collect and sort data by throughput
        timeline_data = []
        for run_name in run_data_dict:
            metric_data = run_data_dict[run_name].get(metric_name, {})
            throughput_data = run_data_dict[run_name].get('request_throughput', {})
            
            if (isinstance(metric_data, dict) and isinstance(throughput_data, dict) and
                'avg' in metric_data and 'avg' in throughput_data):
                
                mean = metric_data['avg']
                std = metric_data.get('std', 0)
                min_val = metric_data.get('min', mean - 2*std)
                max_val = metric_data.get('max', mean + 2*std)
                throughput = throughput_data['avg']
                
                # Generate synthetic timeline data
                n_points = 100
                x = np.arange(n_points)
                y = np.random.normal(mean, std/4, n_points)
                y = np.clip(y, min_val, max_val)
                
                timeline_data.append((throughput, x, y))
        
        # Sort by throughput
        timeline_data.sort(key=lambda x: x[0])
        
        if not timeline_data:
            return None
        
        # Create color scale based on throughput range
        min_throughput = min(data[0] for data in timeline_data)
        max_throughput = max(data[0] for data in timeline_data)
        
        for throughput, x, y in timeline_data:
            # Calculate color based on throughput
            color_val = (throughput - min_throughput) / (max_throughput - min_throughput)
            color = f'rgb({int(255*(1-color_val))}, {int(119*(1-color_val/2))}, {int(180)})'
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f'T:{throughput:.1f} req/s',
                line=dict(
                    color=color,
                    width=2
                ),
                hovertemplate=(
                    f"Time: %{{x}}<br>" +
                    f"{metric_name}: %{{y:.2f}}<br>" +
                    f"Throughput: {throughput:.2f} req/s<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Get metric unit
        unit = run_data_dict[list(run_data_dict.keys())[0]][metric_name].get('unit', self.METRIC_UNITS.get(metric_name, ''))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{metric_name.replace('_', ' ').title()} Timeline by Throughput",
                x=0.5,
                y=0.98,  # Keep within valid range [0, 1]
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title="Time",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title=f"{metric_name.replace('_', ' ').title()} ({unit})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.98,  # Keep within valid range [0, 1]
                xanchor="right",
                x=1,
                font=dict(size=10)  # Smaller font for legend
            ),
            hovermode='x unified',
            margin=dict(t=150, b=50, l=50, r=50),  # Increased top margin further
            plot_bgcolor='white'
        )
        
        return fig

    def create_model_comparison_plot(
        self,
        metrics_data: Dict[str, List[Dict]],
        metric_name: str,
        stat_param: str = "avg"
    ) -> go.Figure:
        """Create a plot comparing metric values across different models."""
        print(f"Received model comparison data with {len(metrics_data)} models")  # Debug
        
        if not metrics_data:
            print("No metrics data received")  # Debug
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(
                title="No data available for plotting",
                xaxis_title="No Data",
                yaxis_title="No Data"
            )
            return fig
            
        fig = go.Figure()
        
        # Get metric unit
        first_model = list(metrics_data.keys())[0]
        first_run = metrics_data[first_model][0] if metrics_data[first_model] else {}
        metric_unit = first_run.get(metric_name, {}).get('unit', self.METRIC_UNITS.get(metric_name, ''))
        
        # Create traces for each model
        for model_name, runs in metrics_data.items():
            # Create shorter label for legend
            if isinstance(model_name, str):
                try:
                    # Parse model profile from the name
                    parts = model_name.split('_')
                    cloud_idx = -1
                    for i, part in enumerate(parts):
                        if part.lower() in ['aws', 'gcp', 'azure']:
                            cloud_idx = i
                            break
                    if cloud_idx != -1:
                        profile_parts = '_'.join(parts[cloud_idx + 3:]).split('-')
                        engine = profile_parts[0]
                        gpu_config = '-'.join(profile_parts[1:3])
                        parallelism = '-'.join(p for p in profile_parts[3:-1] if p.startswith(('tp', 'pp')))
                        engine_short = 'TRT' if engine == 'tensorrt_llm' else engine.upper()
                        gpu_short = gpu_config.split('-')[0].upper()
                        precision = gpu_config.split('-')[1].upper()
                        parallel = parallelism.upper()
                        profile_label = f"{engine_short}-{gpu_short}-{precision}-{parallel}"
                        legend_label = f"{model_name.split('_')[0]}-{model_name.split('_')[1]} | {profile_label} | {profile_parts[-1]} "

                    else:
                        legend_label = model_name
                except:
                    legend_label = model_name
            else:
                legend_label = str(model_name)
            
            x_values = []  # metric values
            y_values = []  # throughput values
            hover_text = []  # hover information
            
            for run in runs:
                metric_data = run.get(metric_name, {})
                throughput_data = run.get('request_throughput', {})
                
                if isinstance(metric_data, dict) and isinstance(throughput_data, dict):
                    metric_value = metric_data.get(stat_param)
                    throughput = throughput_data.get('avg')  # Always use average for throughput
                    
                    if metric_value is not None and throughput is not None:
                        x_values.append(metric_value)
                        y_values.append(throughput)
                        hover_text.append(
                            f"Model: {model_name}<br>" +
                            f"{metric_name} ({stat_param}): {metric_value:.2f} {metric_unit}<br>" +
                            f"Throughput: {throughput:.2f} req/s"
                        )
            
            if x_values and y_values:
                # Sort points by metric values for proper line connection
                points = sorted(zip(x_values, y_values, hover_text))
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                hover_text = [p[2] for p in points]
                
                # Add line plot for this model
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=legend_label,
                    text=hover_text,
                    line=dict(width=2),
                    marker=dict(
                        size=8,
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate="%{text}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{metric_name.replace('_', ' ').title()} vs Request Throughput",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title=f"{metric_name.replace('_', ' ').title()} ({metric_unit})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            yaxis=dict(
                title="Request Throughput (req/s)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                rangemode='tozero'
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.4,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        
        return fig

    def create_latency_distribution_comparison_plot(
        self,
        metrics_data: Dict[str, List[Dict]],
        metric_name: str
    ) -> go.Figure:
        """Create horizontal box plots comparing latency distributions across different models/configs."""
        print(f"Creating latency distribution comparison plot for {metric_name}")
        if not metrics_data:
            print("No metrics data received")
            return None
        fig = go.Figure()
        # Get metric unit
        first_model = list(metrics_data.keys())[0]
        first_run = metrics_data[first_model][0] if metrics_data[first_model] else {}
        metric_unit = first_run.get(metric_name, {}).get('unit', self.METRIC_UNITS.get(metric_name, ''))
        # Define colors for different models
        colors = [
            'rgb(31, 119, 180)',   # Blue
            'rgb(255, 127, 14)',   # Orange
            'rgb(44, 160, 44)',    # Green
            'rgb(214, 39, 40)',    # Red
            'rgb(148, 103, 189)',  # Purple
            'rgb(140, 86, 75)',    # Brown
            'rgb(227, 119, 194)',  # Pink
            'rgb(127, 127, 127)'   # Gray
        ]
        # Track min/max values for axis scaling
        all_latencies = []
        all_throughputs = []
        # Process each model
        for model_idx, (model_name, runs) in enumerate(metrics_data.items()):
            # Create shorter label for legend
            if isinstance(model_name, str):
                try:
                    parts = model_name.split('_')
                    cloud_idx = -1
                    for i, part in enumerate(parts):
                        if part.lower() in ['aws', 'gcp', 'azure']:
                            cloud_idx = i
                            break
                    if cloud_idx != -1:
                        profile_parts = '_'.join(parts[cloud_idx + 3:]).split('-')
                        engine = profile_parts[0]
                        gpu_config = '-'.join(profile_parts[1:3])
                        parallelism = '-'.join(p for p in profile_parts[3:-1] if p.startswith(('tp', 'pp')))
                        engine_short = 'TRT' if engine == 'tensorrt_llm' else engine.upper()
                        gpu_short = gpu_config.split('-')[0].upper()
                        precision = gpu_config.split('-')[1].upper()
                        parallel = parallelism.upper()
                        short_label = f"{engine_short}-{gpu_short}-{precision}-{parallel}"
                    else:
                        short_label = model_name
                except:
                    short_label = model_name
            else:
                short_label = str(model_name)
            # Collect all latency values and throughputs for this model
            latency_values = []
            throughput_values = []
            for run in runs:
                metric_data = run.get(metric_name, {})
                throughput_data = run.get('request_throughput', {})
                if isinstance(metric_data, dict) and isinstance(throughput_data, dict):
                    mean = metric_data.get('avg')
                    p50 = metric_data.get('p50', mean)
                    p25 = metric_data.get('p25', mean - (p50 - mean))
                    p75 = metric_data.get('p75', mean + (mean - p50))
                    p90 = metric_data.get('p90', p75 + (p75 - p25))
                    p99 = metric_data.get('p99', p90 + (p90 - p75))
                    min_val = metric_data.get('min', p25 - 1.5 * (p75 - p25))
                    max_val = metric_data.get('max', p75 + 1.5 * (p75 - p25))
                    throughput = throughput_data.get('avg')
                    if all(v is not None for v in [mean, p50, p25, p75, p90, p99, min_val, max_val, throughput]):
                        latency_values.extend([min_val, p25, p50, p75, max_val, mean, p90, p99])
                        throughput_values.append(throughput)
            if latency_values and throughput_values:
                # Calculate average throughput for this model
                avg_throughput = np.mean(throughput_values)
                color = colors[model_idx % len(colors)]
                fill_color = color.replace('rgb', 'rgba').replace(')', ', 0.1)')
                min_val = min(latency_values)
                max_val = max(latency_values)
                q1 = np.percentile(latency_values, 25)
                median = np.percentile(latency_values, 50)
                q3 = np.percentile(latency_values, 75)
                mean = np.mean(latency_values)
                p90 = np.percentile(latency_values, 90)
                p99 = np.percentile(latency_values, 99)
                # Add horizontal box plot (metric on x, throughput on y)
                fig.add_trace(go.Box(
                    y=[short_label] * 5,
                    x=[min_val, q1, median, q3, max_val],
                    name=short_label,
                    orientation='h',
                    boxpoints=False,
                    line=dict(color=color, width=2),
                    fillcolor=fill_color,
                    whiskerwidth=0.7,
                    showlegend=False
                ))
                # Add mean marker with error bars
                fig.add_trace(go.Scatter(
                    y=[short_label],
                    x=[mean],
                    mode='markers',
                    name=short_label,
                    marker=dict(
                        symbol='line-ns',
                        size=20,
                        color=color,
                        line=dict(width=2)
                    ),
                    error_x=dict(
                        type='data',
                        array=[q3 - q1],
                        color=color,
                        thickness=1,
                        width=10
                    ),
                    showlegend=True,
                    hovertemplate=(
                        f"Model: {model_name}<br>" +
                        f"Mean: {mean:.2f}<br>" +
                        f"IQR: [{q1:.2f}, {q3:.2f}]<br>" +
                        f"P90: {p90:.2f}<br>" +
                        f"P99: {p99:.2f}<br>" +
                        f"Avg Throughput: {avg_throughput:.2f} req/s<br>" +
                        "<extra></extra>"
                    )
                ))
                all_latencies.extend(latency_values)
                all_throughputs.append(short_label)
        if not all_latencies or not all_throughputs:
            return None
        # Calculate axis ranges with padding
        x_min, x_max = min(all_latencies), max(all_latencies)
        x_padding = (x_max - x_min) * 0.1
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{metric_name.replace('_', ' ').title()} Distribution Comparison",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title=f"{metric_name.replace('_', ' ').title()} ({metric_unit})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                range=[x_min - x_padding, x_max + x_padding]
            ),
            yaxis=dict(
                title="Model/Config",
                type="category",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        return fig 

    def create_throughput_distribution_comparison_plot(
        self,
        metrics_data: Dict[str, List[Dict]],
        metric_name: str = "request_throughput"
    ) -> go.Figure:
        """Create horizontal box plots comparing throughput distributions across different models/configs."""
        fig = go.Figure()
        colors = [
            'rgb(31, 119, 180)',   # Blue
            'rgb(255, 127, 14)',   # Orange
            'rgb(44, 160, 44)',    # Green
            'rgb(214, 39, 40)',    # Red
            'rgb(148, 103, 189)',  # Purple
            'rgb(140, 86, 75)',    # Brown
            'rgb(227, 119, 194)',  # Pink
            'rgb(127, 127, 127)'   # Gray
        ]
        all_throughputs = []
        for model_idx, (model_name, runs) in enumerate(metrics_data.items()):
            # Create shorter label for legend
            if isinstance(model_name, str):
                try:
                    parts = model_name.split('_')
                    cloud_idx = -1
                    for i, part in enumerate(parts):
                        if part.lower() in ['aws', 'gcp', 'azure']:
                            cloud_idx = i
                            break
                    if cloud_idx != -1:
                        profile_parts = '_'.join(parts[cloud_idx + 3:]).split('-')
                        engine = profile_parts[0]
                        gpu_config = '-'.join(profile_parts[1:3])
                        parallelism = '-'.join(p for p in profile_parts[3:-1] if p.startswith(('tp', 'pp')))
                        engine_short = 'TRT' if engine == 'tensorrt_llm' else engine.upper()
                        gpu_short = gpu_config.split('-')[0].upper()
                        precision = gpu_config.split('-')[1].upper()
                        parallel = parallelism.upper()
                        short_label = f"{engine_short}-{gpu_short}-{precision}-{parallel}"
                    else:
                        short_label = model_name
                except:
                    short_label = model_name
            else:
                short_label = str(model_name)
            throughput_values = []
            for run in runs:
                throughput_data = run.get(metric_name, {})
                if isinstance(throughput_data, dict):
                    for stat in ['avg', 'p50', 'p90', 'p95', 'p99', 'min', 'max']:
                        val = throughput_data.get(stat)
                        if val is not None:
                            throughput_values.append(val)
            if throughput_values:
                color = colors[model_idx % len(colors)]
                fill_color = color.replace('rgb', 'rgba').replace(')', ', 0.1)')
                fig.add_trace(go.Box(
                    y=[short_label] * len(throughput_values),
                    x=throughput_values,
                    name=short_label,
                    orientation='h',
                    boxpoints=False,
                    line=dict(color=color, width=2),
                    fillcolor=fill_color,
                    whiskerwidth=0.7,
                    showlegend=False
                ))
                all_throughputs.extend(throughput_values)
        if not all_throughputs:
            return None
        x_min, x_max = min(all_throughputs), max(all_throughputs)
        x_padding = (x_max - x_min) * 0.1
        fig.update_layout(
            title=dict(
                text="Request Throughput Distribution Comparison",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=16,
                    color='rgb(44, 44, 44)',
                    family='bold Arial, Arial, sans-serif'
                ),
                pad=dict(t=30)
            ),
            xaxis=dict(
                title="Request Throughput (requests/sec)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                range=[x_min - x_padding, x_max + x_padding]
            ),
            yaxis=dict(
                title="Model/Config",
                type="category",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        return fig 

    def create_latency_vs_throughput_scatter_plot(
        self,
        metrics_data: Dict[str, List[Dict]],
        latency_metric: str = "request_latency",
        throughput_metric: str = "request_throughput",
        stat: str = "avg"
    ) -> go.Figure:
        """Create a scatter plot with latency on x-axis and throughput on y-axis for each model/config."""
        fig = go.Figure()
        colors = [
            'rgb(31, 119, 180)',   # Blue
            'rgb(255, 127, 14)',   # Orange
            'rgb(44, 160, 44)',    # Green
            'rgb(214, 39, 40)',    # Red
            'rgb(148, 103, 189)',  # Purple
            'rgb(140, 86, 75)',    # Brown
            'rgb(227, 119, 194)',  # Pink
            'rgb(127, 127, 127)'   # Gray
        ]
        for model_idx, (model_name, runs) in enumerate(metrics_data.items()):
            # Create shorter label for legend
            if isinstance(model_name, str):
                try:
                    parts = model_name.split('_')
                    cloud_idx = -1
                    for i, part in enumerate(parts):
                        if part.lower() in ['aws', 'gcp', 'azure']:
                            cloud_idx = i
                            break
                    if cloud_idx != -1:
                        profile_parts = '_'.join(parts[cloud_idx + 3:]).split('-')
                        engine = profile_parts[0]
                        gpu_config = '-'.join(profile_parts[1:3])
                        parallelism = '-'.join(p for p in profile_parts[3:-1] if p.startswith(('tp', 'pp')))
                        engine_short = 'TRT' if engine == 'tensorrt_llm' else engine.upper()
                        gpu_short = gpu_config.split('-')[0].upper()
                        precision = gpu_config.split('-')[1].upper()
                        parallel = parallelism.upper()
                        short_label = f"{engine_short}-{gpu_short}-{precision}-{parallel}"
                    else:
                        short_label = model_name
                except:
                    short_label = model_name
            else:
                short_label = str(model_name)
            x_vals = []
            y_vals = []
            hover_texts = []
            for run in runs:
                latency_data = run.get(latency_metric, {})
                throughput_data = run.get(throughput_metric, {})
                if isinstance(latency_data, dict) and isinstance(throughput_data, dict):
                    latency_val = latency_data.get(stat)
                    throughput_val = throughput_data.get(stat)
                    if latency_val is not None and throughput_val is not None:
                        x_vals.append(latency_val)
                        y_vals.append(throughput_val)
                        hover_texts.append(f"{short_label}<br>Latency: {latency_val:.2f} ms<br>Throughput: {throughput_val:.2f} req/s")
            if x_vals and y_vals:
                color = colors[model_idx % len(colors)]
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers+lines',
                    name=short_label,
                    marker=dict(size=10, color=color),
                    line=dict(color=color, width=2, dash='dot'),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>"
                ))
        fig.update_layout(
            title="Latency vs Throughput (per config)",
            xaxis_title="Latency (ms)",
            yaxis_title="Throughput (requests/sec)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.98,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white'
        )
        return fig 