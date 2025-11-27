"""
IMDB Dataset Inspection Script

This script performs exploratory data analysis on the IMDB movie reviews dataset:
- Class balance analysis
- Review length statistics
- Data quality checks
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the IMDB dataset from CSV file."""
    logger.info(f'Loading data from {data_path}...')
    df = pd.read_csv(data_path)
    logger.success(f'Dataset loaded successfully. Shape: {df.shape}')
    return df


def inspect_class_balance(
    df: pd.DataFrame, output_dir: Path | None = None, show_plot: bool = True
) -> None:
    """Analyze and visualize class balance in the dataset.

    Args:
        df: DataFrame containing the IMDB reviews data
        output_dir: Directory to save visualization files. If None, saves to current directory.
        show_plot: Whether to display plots interactively. If False, only saves to file.
    """
    logger.info('\n' + '=' * 60)
    logger.info('CLASS BALANCE ANALYSIS')
    logger.info('=' * 60)

    # Count by sentiment
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100

    logger.info('\nSentiment Distribution:')
    for sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        percentage = sentiment_percentages[sentiment]
        logger.info(f'  {sentiment}: {count:,} ({percentage:.2f}%)')

    # Visualize class balance
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    sentiment_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Sentiment Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentiment', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)

    # Add count labels on bars
    for i, (_sentiment, count) in enumerate(sentiment_counts.items()):
        axes[0].text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=11)

    # Pie chart
    colors = (
        ['#2ecc71', '#e74c3c']
        if sentiment_counts.index[0] == 'positive'
        else ['#e74c3c', '#2ecc71']
    )
    axes[1].pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct='%1.2f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12},
    )
    axes[1].set_title(
        'Sentiment Distribution (Percentage)', fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    if output_dir is None:
        output_file = Path('class_balance.png')
    else:
        output_file = output_dir / 'class_balance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"Class balance visualization saved as '{output_file}'")

    if show_plot:
        plt.show()
    else:
        plt.close()


def inspect_review_lengths(
    df: pd.DataFrame, output_dir: Path | None = None, show_plot: bool = True
) -> None:
    """Analyze and visualize review length statistics.

    Args:
        df: DataFrame containing the IMDB reviews data
        output_dir: Directory to save visualization files. If None, saves to current directory.
        show_plot: Whether to display plots interactively. If False, only saves to file.
    """
    logger.info('\n' + '=' * 60)
    logger.info('REVIEW LENGTH ANALYSIS')
    logger.info('=' * 60)

    # Calculate review lengths
    df['review_length'] = df['review'].str.len()
    df['word_count'] = df['review'].str.split().str.len()

    # Overall statistics
    logger.info('\nOverall Statistics:')
    logger.info(
        f'  Character count - Mean: {df["review_length"].mean():.2f}, '
        f'Median: {df["review_length"].median():.0f}, '
        f'Std: {df["review_length"].std():.2f}'
    )
    logger.info(
        f'  Word count      - Mean: {df["word_count"].mean():.2f}, '
        f'Median: {df["word_count"].median():.0f}, '
        f'Std: {df["word_count"].std():.2f}'
    )
    logger.info(
        f'  Min words: {df["word_count"].min()}, Max words: {df["word_count"].max()}'
    )

    # Statistics by sentiment
    logger.info('\nStatistics by Sentiment:')
    for sentiment in df['sentiment'].unique():
        sentiment_df = df[df['sentiment'] == sentiment]
        logger.info(f'\n  {sentiment.upper()}:')
        logger.info(
            f'    Character count - Mean: {sentiment_df["review_length"].mean():.2f}, '
            f'Median: {sentiment_df["review_length"].median():.0f}'
        )
        logger.info(
            f'    Word count      - Mean: {sentiment_df["word_count"].mean():.2f}, '
            f'Median: {sentiment_df["word_count"].median():.0f}'
        )

    # Detailed percentile statistics
    logger.info('\nPercentile Statistics (Word Count):')
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = df['word_count'].quantile([p / 100 for p in percentiles])
    for p, val in zip(percentiles, percentile_values, strict=True):
        logger.info(f'  {p}th percentile: {val:.0f} words')

    # Visualizations
    _fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Distribution of word counts (histogram)
    axes[0, 0].hist(
        df['word_count'], bins=50, color='steelblue', edgecolor='black', alpha=0.7
    )
    axes[0, 0].set_title(
        'Distribution of Review Word Counts', fontsize=14, fontweight='bold'
    )
    axes[0, 0].set_xlabel('Word Count', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].axvline(
        df['word_count'].mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {df["word_count"].mean():.0f}',
    )
    axes[0, 0].axvline(
        df['word_count'].median(),
        color='green',
        linestyle='--',
        linewidth=2,
        label=f'Median: {df["word_count"].median():.0f}',
    )
    axes[0, 0].legend()

    # 2. Box plot by sentiment
    df.boxplot(column='word_count', by='sentiment', ax=axes[0, 1])
    axes[0, 1].set_title(
        'Word Count Distribution by Sentiment', fontsize=14, fontweight='bold'
    )
    axes[0, 1].set_xlabel('Sentiment', fontsize=12)
    axes[0, 1].set_ylabel('Word Count', fontsize=12)
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)

    # 3. Violin plot by sentiment
    sentiment_order = sorted(df['sentiment'].unique())
    sns.violinplot(
        data=df, x='sentiment', y='word_count', ax=axes[1, 0], order=sentiment_order
    )
    axes[1, 0].set_title(
        'Word Count Distribution by Sentiment (Violin Plot)',
        fontsize=14,
        fontweight='bold',
    )
    axes[1, 0].set_xlabel('Sentiment', fontsize=12)
    axes[1, 0].set_ylabel('Word Count', fontsize=12)
    axes[1, 0].set_ylim(
        0, df['word_count'].quantile(0.99) * 1.1
    )  # Limit to 99th percentile for better visualization

    # 4. Cumulative distribution
    for sentiment in sentiment_order:
        sentiment_data = df[df['sentiment'] == sentiment]['word_count'].sort_values()
        cumulative = range(1, len(sentiment_data) + 1)
        axes[1, 1].plot(sentiment_data.values, cumulative, label=sentiment, linewidth=2)
    axes[1, 1].set_title(
        'Cumulative Distribution of Word Counts', fontsize=14, fontweight='bold'
    )
    axes[1, 1].set_xlabel('Word Count', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative Count', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir is None:
        output_file = Path('review_lengths.png')
    else:
        output_file = output_dir / 'review_lengths.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.success(f"Review length visualization saved as '{output_file}'")

    if show_plot:
        plt.show()
    else:
        plt.close()


def inspect_data_quality(df: pd.DataFrame) -> None:
    """Check for data quality issues."""
    logger.info('\n' + '=' * 60)
    logger.info('DATA QUALITY CHECKS')
    logger.info('=' * 60)

    # Missing values
    logger.info('\nMissing Values:')
    missing = df.isnull().sum()
    if missing.sum() == 0:
        logger.success('  No missing values found')
    else:
        logger.warning(missing[missing > 0])

    # Duplicate reviews
    duplicates = df.duplicated(subset=['review']).sum()
    logger.info(f'\nDuplicate Reviews: {duplicates}')
    if duplicates > 0:
        logger.warning(
            f'  Found {duplicates} duplicate reviews ({duplicates / len(df) * 100:.2f}%)'
        )
    else:
        logger.success('  No duplicate reviews found')

    # Empty or very short reviews
    min_review_length = 10
    short_reviews = (df['review'].str.len() < min_review_length).sum()
    logger.info(
        f'\nVery Short Reviews (< {min_review_length} characters): {short_reviews}'
    )
    if short_reviews > 0:
        logger.warning(f'  Found {short_reviews} very short reviews')
    else:
        logger.success('  No suspiciously short reviews')

    # Unique sentiment values
    logger.info(f'\nUnique Sentiment Values: {df["sentiment"].unique().tolist()}')


@click.command()
@click.option(
    '--data-path',
    '-d',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to the IMDB dataset CSV file. If not provided, uses default location.',
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(path_type=Path),
    default='artifacts',
    help='Directory to save visualization files. Default is `artifacts` directory.',
)
@click.option(
    '--no-show',
    is_flag=True,
    help='Do not display plots interactively (only save to files).',
)
@click.option(
    '--skip-quality',
    is_flag=True,
    help='Skip data quality checks.',
)
@click.option(
    '--skip-balance',
    is_flag=True,
    help='Skip class balance analysis.',
)
@click.option(
    '--skip-lengths',
    is_flag=True,
    help='Skip review length analysis.',
)
def inspect_dataset(
    data_path: Path | None,
    output_dir: Path,
    no_show: bool,
    skip_quality: bool,
    skip_balance: bool,
    skip_lengths: bool,
):
    """Inspect IMDB movie reviews dataset.

    Performs comprehensive analysis including:
    - Data quality checks (missing values, duplicates, short reviews)
    - Class balance analysis with visualizations
    - Review length statistics and distributions

    Examples:

        \b
        # Use default data location
        python inspection.py

        \b
        # Specify custom data file
        python inspection.py -d /path/to/data.csv

        \b
        # Save visualizations to specific directory without showing
        python inspection.py -o ./reports --no-show

        \b
        # Run only specific analyses
        python inspection.py --skip-quality --skip-lengths
    """
    # Determine data path
    if data_path is None:
        data_path = Path(__file__).parents[2] / 'data' / 'raw' / 'IMDB Dataset.csv'
        logger.info(f'Using default data path: {data_path}')

    # Check if file exists
    if not data_path.exists():
        logger.error(f'Data file not found at {data_path}')
        raise click.Abort

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Output directory: {output_dir}')

    # Configure matplotlib backend if not showing plots
    if no_show:
        plt.ioff()
        logger.info('Interactive plotting disabled')

    # Load data
    df = load_data(data_path)

    # Perform inspections based on flags
    analyses_run = []

    if not skip_quality:
        inspect_data_quality(df)
        analyses_run.append('Data Quality')

    if not skip_balance:
        inspect_class_balance(df, output_dir, show_plot=not no_show)
        analyses_run.append('Class Balance')

    if not skip_lengths:
        inspect_review_lengths(df, output_dir, show_plot=not no_show)
        analyses_run.append('Review Lengths')

    # Summary
    logger.info('\n' + '=' * 60)
    logger.success('INSPECTION COMPLETE')
    logger.info('=' * 60)
    logger.info(f'\nAnalyses performed: {", ".join(analyses_run)}')

    if not skip_balance or not skip_lengths:
        logger.info(f'\nVisualization files saved to: {output_dir.absolute()}')
        if not skip_balance:
            logger.info(f'  - {output_dir / "class_balance.png"}')
        if not skip_lengths:
            logger.info(f'  - {output_dir / "review_lengths.png"}')


if __name__ == '__main__':
    inspect_dataset()
