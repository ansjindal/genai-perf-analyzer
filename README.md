# GenAI Performance Analyzer

A streamlined tool for analyzing and visualizing performance metrics of Large Language Models (LLMs) across different configurations and runtime environments.

## Features

- **Interactive Visualization**: Dynamic plots for latency and throughput metrics
- **Configuration Comparison**: Compare different model configurations side by side
- **Metric Analysis**: Analyze various performance metrics including:
  - Request Latency
  - Time to First Token
  - Inter-token Latency
  - Request Throughput
  - Output Token Throughput
  - Output Token Throughput per Request

## Architecture

The application follows a modular architecture with clear separation of concerns. View the detailed [Architecture Documentation](docs/architecture.md) for:
- System Components Overview
- Data Flow Diagram
- Component Descriptions
- Technology Stack Details

## Interface Preview

<details>
<summary>Click to view interface screenshots</summary>

### Latency Distribution View
![Latency Distribution](docs/images/latency_distribution.png)
*Analyze latency distributions across different token configurations with interactive box plots and statistical insights*

### Model Comparison View
![Model Comparison](docs/images/model_comparison.png)
*Compare performance metrics between different model configurations and concurrency levels*

### Overall Interface
![Overall UI](docs/images/overall_ui.png)
*Complete interface with sidebar controls and metric visualization panels*

</details>

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genai-perf-analyzer.git
cd genai-perf-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your performance data in the `data/` directory following the expected format:
```
data/
  ├── model_name_provider_instance/
  │   └── model_name_concurrencyN/
  │       └── token_config.json
```

2. Run the Streamlit application:
```bash
streamlit run app/main.py
```

3. Access the web interface at `http://localhost:8501`

## Data Format

The analyzer expects performance data in JSON format with the following structure:

```json
{
  "request_latency": {
    "unit": "ms",
    "avg": 100.0,
    "p50": 95.0,
    "p90": 150.0,
    "p95": 180.0,
    "p99": 200.0,
    "min": 50.0,
    "max": 250.0,
    "std": 30.0
  },
  // ... other metrics
}
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualization powered by [Plotly](https://plotly.com/)
- Data analysis using [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/)