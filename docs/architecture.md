# Architecture Overview

## System Components

```mermaid
graph TB
    subgraph Data Layer
        D[Data Directory] --> |JSON Files| DL[Data Loader]
        DL --> |Metrics & Stats| DC[Data Cache]
    end

    subgraph Application Core
        DC --> |Processed Data| V[Visualizer]
        V --> |Plot Objects| UI[UI Components]
    end

    subgraph Web Interface
        UI --> |Latency Metrics| LM[Latency Module]
        UI --> |Throughput Metrics| TM[Throughput Module]
        
        LM --> |Plots| P1[Distribution Plots]
        LM --> |Plots| P2[Timeline Plots]
        TM --> |Plots| P3[Comparison Plots]
        TM --> |Plots| P4[Concurrency Plots]
    end

    subgraph User Interaction
        C[Configuration Selection] --> |User Input| UI
        MT[Metric Type Selection] --> |User Input| UI
        PT[Plot Type Selection] --> |User Input| UI
    end

    style Data Layer fill:#e1f5fe,stroke:#01579b
    style Application Core fill:#f3e5f5,stroke:#4a148c
    style Web Interface fill:#e8f5e9,stroke:#1b5e20
    style User Interaction fill:#fff3e0,stroke:#e65100
```

## Component Description

### Data Layer
- **Data Directory**: Contains performance metrics in JSON format
- **Data Loader**: Processes JSON files and extracts metrics
- **Data Cache**: Stores processed metrics for quick access

### Application Core
- **Visualizer**: Creates interactive plots using Plotly
- **UI Components**: Streamlit components for web interface

### Web Interface
- **Latency Module**: Handles latency-related visualizations
- **Throughput Module**: Handles throughput-related visualizations
- **Plot Types**:
  - Distribution Plots: Box plots for metric distributions
  - Timeline Plots: Time series visualization
  - Comparison Plots: Bar charts for metric comparisons
  - Concurrency Plots: Line plots for concurrency analysis

### User Interaction
- **Configuration Selection**: Model and token config selection
- **Metric Type Selection**: Choose metrics to analyze
- **Plot Type Selection**: Select visualization type

## Data Flow

1. User selects configurations and metrics through the web interface
2. Data Loader reads and processes JSON files from the data directory
3. Processed data is cached for performance
4. Visualizer creates appropriate plots based on user selection
5. UI components render the plots and handle user interaction

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: NumPy, Pandas
- **File Handling**: Python standard library 