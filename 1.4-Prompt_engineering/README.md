# Streamlit Marketing Mix Modeling (MMM) App

A production-ready Marketing Mix Modeling application built with Streamlit for analyzing the effectiveness of marketing channels on sales performance.

## Features

- Interactive data upload and validation
- Multi-channel attribution analysis (TV, Radio, Digital)
- Visual insights with interactive charts
- Sales contribution breakdown by marketing channel

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

### 1. Install uv (if not already installed)

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone or navigate to the project directory

```bash
cd path/to/streamlit-mmm-app
```

### 3. Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs all required packages.

## How to Run

### Start the Streamlit app:

```bash
uv run streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

### Development mode with auto-reload:

```bash
uv run streamlit run app.py --server.runOnSave true
```

## Expected CSV Data Schema

The application expects a CSV file with the following structure:

| Column Name    | Data Type | Description                          | Example      |
|----------------|-----------|--------------------------------------|--------------|
| `date`         | Date      | Date of observation (YYYY-MM-DD)     | 2024-01-01   |
| `Sales`        | Float     | Total sales revenue                  | 125000.50    |
| `TV_Spend`     | Float     | Television advertising spend         | 15000.00     |
| `Radio_Spend`  | Float     | Radio advertising spend              | 8000.00      |
| `Digital_Spend`| Float     | Digital marketing spend              | 12000.00     |

### Example CSV Format:

```csv
date,Sales,TV_Spend,Radio_Spend,Digital_Spend
2024-01-01,125000.50,15000.00,8000.00,12000.00
2024-01-08,130000.75,16000.00,8500.00,13000.00
2024-01-15,128000.00,15500.00,8200.00,12500.00
```

### Data Requirements:

- **Date format**: ISO format (YYYY-MM-DD) or common formats (MM/DD/YYYY, DD-MM-YYYY)
- **Numeric values**: All spend and sales columns must be numeric (integers or floats)
- **No missing values**: Ensure all required columns are present without null values
- **Minimum rows**: At least 20 observations recommended for meaningful analysis

## Project Structure

```
.
├── pyproject.toml      # Project dependencies and configuration
├── README.md           # This file
├── app.py              # Main Streamlit application
├── data/               # Sample data directory
│   └── sample_mmm.csv  # Example dataset
└── utils/              # Utility modules
    ├── models.py       # MMM modeling logic
    └── visualizations.py # Chart generation functions
```

## Screenshots

### 1. Data Upload Interface
![Data Upload](screenshots/01_data_upload.png)
*Upload your marketing data CSV file with drag-and-drop functionality*

### 2. Data Preview & Validation
![Data Preview](screenshots/02_data_preview.png)
*Preview uploaded data and validate schema compliance*

### 3. Marketing Channel Performance
![Channel Performance](screenshots/03_channel_performance.png)
*Analyze contribution of each marketing channel to overall sales*

### 4. Time Series Analysis
![Time Series](screenshots/04_time_series.png)
*Visualize sales trends alongside marketing spend over time*

### 5. Attribution Results
![Attribution](screenshots/05_attribution_results.png)
*View detailed attribution metrics and ROI by channel*

## Development

### Install development dependencies:

```bash
uv sync --extra dev
```

### Run tests:

```bash
uv run pytest
```

### Format code:

```bash
uv run black .
uv run ruff check --fix .
```

## Troubleshooting

**Issue**: Port 8501 already in use  
**Solution**: Specify a different port: `uv run streamlit run app.py --server.port 8502`

**Issue**: CSV upload fails  
**Solution**: Verify CSV matches the expected schema exactly (column names are case-sensitive)

**Issue**: Virtual environment not found  
**Solution**: Run `uv sync` again to recreate the environment

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open an issue in the project repository.
