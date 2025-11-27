# LSTM Sentiment Classification

A comprehensive toolkit for sentiment analysis on IMDB movie reviews using LSTM neural networks.

## Installation

### Using UV (Recommended)

```bash
# Install dependencies with uv
uv sync

# The CLI will be available as 'sentiment-cli'
sentiment-cli --help
```

### Using pip

```bash
# Install dependencies
pip install click pandas matplotlib seaborn loguru

# Install in development mode
pip install -e .

# Or use the package directly
python -m sentiment_classification --help
```

## CLI Usage

The project provides a command-line interface organized into subgroups.

### Main CLI

```bash
# Using the installed script (after uv sync)
sentiment-cli --help
sentiment-cli --version

# Or using Python module
python -m sentiment_classification --help
python -m sentiment_classification --version
```

### Data Commands

Commands for data inspection, preprocessing, and management.

```bash
# Show data subcommands
sentiment-cli data --help

# Inspect dataset (basic usage)
sentiment-cli data inspect

# Inspect with custom data file
sentiment-cli data inspect -d path/to/data.csv

# Save visualizations to specific directory
sentiment-cli data inspect -o ./reports

# Skip interactive plots (save only)
sentiment-cli data inspect --no-show

# Run specific analyses only
sentiment-cli data inspect --skip-quality
sentiment-cli data inspect --skip-balance --skip-lengths
```

**Alternative:** You can also use `python -m sentiment_classification` instead of `sentiment-cli` for all commands.

### Train Commands

Commands for model training (to be implemented).

```bash
# Show train subcommands
sentiment-cli train --help
```

### Eval Commands

Commands for model evaluation (to be implemented).

```bash
# Show eval subcommands
sentiment-cli eval --help
```

## Project Structure

```
sentiment_classification/
├── __init__.py
├── __main__.py          # Package entry point
├── cli.py               # Main CLI with groups
└── workflows/
    ├── __init__.py
    └── inspection.py    # Data inspection functionality
data/
└── raw/
    └── IMDB Dataset.csv
```

## Data Inspection

The `data inspect` command performs comprehensive analysis:

- **Data Quality Checks**: Missing values, duplicates, short reviews
- **Class Balance Analysis**: Distribution visualizations (bar plot, pie chart)
- **Review Length Statistics**: Word count distributions, percentiles, multiple visualizations

### Output

Generated visualizations are saved to the specified output directory (default: `artifacts/`):
- `class_balance.png`: Sentiment distribution charts
- `review_lengths.png`: Word count analysis plots

## Development

```bash
# Install in development mode with uv
uv sync

# Run CLI
sentiment-cli --help

# Or using Python module
python -m sentiment_classification

# Run specific workflow directly
python sentiment_classification/workflows/inspection.py --help
```

## Quick Start

```bash
# 1. Install with uv
uv sync

# 2. Inspect the dataset
sentiment-cli data inspect

# 3. View help for all commands
sentiment-cli --help
sentiment-cli data --help
sentiment-cli train --help
sentiment-cli eval --help
```
