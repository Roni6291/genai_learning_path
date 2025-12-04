"""
Main execution script for SQL query generation
Generates SQL queries for all 20 business questions and saves results to CSV
"""

import csv
import json

from loguru import logger

from sql_generator import SQLQueryGenerator


def load_questions(filepath="questions.json"):
    """Load business questions from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']


def save_to_csv(results, output_file="dwbi_sql_output.csv"):
    """
    Save results to CSV file with required columns
    
    Args:
        results (list): List of query results
        output_file (str): Output CSV file path
    """
    # Define CSV columns in exact order
    fieldnames = ['question_id', 'question', 'target_source', 'sql', 'assumptions', 'confidence']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'question_id': result['question_id'],
                'question': result['question'],
                'target_source': result['target_source'],
                'sql': result['sql'],
                'assumptions': result['assumptions'],
                'confidence': result['confidence']
            })


def main():
    """
    Main execution function
    """
    logger.info("=" * 80)
    logger.info("SQL Query Generator for DWBI Team")
    logger.info("Using Groq API (Free Tier Optimized)")
    logger.info("=" * 80)
    
    # Initialize the SQL generator with llama-3.1-8b-instant model (free tier optimized)
    # You can change the model here if needed:
    # Free tier options: llama-3.1-8b-instant, meta-llama/llama-4-scout-17b-16e-instruct, qwen/qwen3-32b
    try:
        generator = SQLQueryGenerator(model_name="llama-3.1-8b-instant")
        logger.success("Initialized SQL Generator with model: llama-3.1-8b-instant")
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.info("Please follow these steps:")
        logger.info("1. Copy .env.example to .env")
        logger.info("2. Add your Groq API key to the .env file")
        logger.info("3. Run the script again")
        return
    
    # Load questions from JSON file
    business_questions = load_questions()
    
    # Generate queries for all questions
    logger.info(f"Generating SQL queries for {len(business_questions)} business questions...")
    logger.info("-" * 80)
    
    results = generator.generate_queries_batch(business_questions)
    
    logger.info("-" * 80)
    logger.info("Saving results to CSV...")
    
    # Save to CSV file with required format
    csv_filename = "output/dwbi_sql_output.csv"
    save_to_csv(results, csv_filename)
    logger.success(f"Saved results to '{csv_filename}'")
    
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total questions processed: {len(results)}")
    
    logger.success("All queries generated successfully!")
    logger.info(f"Output file: {csv_filename}")
    logger.info("Columns: question_id, question, target_source, sql, assumptions, confidence")


if __name__ == "__main__":
    main()
