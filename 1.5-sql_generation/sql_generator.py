"""
SQL Query Generator using Groq API
This module handles the generation of SQL queries from natural language questions
"""

import os
import json
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
from loguru import logger


def load_text_file(filename):
    """Load content from text file"""
    file_path = Path(__file__).parent / filename
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# Load schemas and prompt template from text files
SALES_DW_SCHEMA = load_text_file("table_schemas/sales_dw_schema.txt")
MARKETING_DW_SCHEMA = load_text_file("table_schemas/marketing_dw_schema.txt")
PROMPT_TEMPLATE = load_text_file("prompt_template.txt")


class SQLQueryGenerator:
    """
    Generates SQL queries using Groq API based on business questions
    """
    
    def __init__(self, model_name="llama-3.1-8b-instant"):
        """
        Initialize the SQL Query Generator
        
        Args:
            model_name (str): Groq model to use for inference
                Free tier recommended options:
                - llama-3.1-8b-instant (default, 14.4K req/day, 500K tokens/day)
                - meta-llama/llama-4-scout-17b-16e-instruct (30K tokens/min, 500K tokens/day)
                - qwen/qwen3-32b (60 req/min, 500K tokens/day)
                - llama-3.3-70b-versatile (1K req/day, 100K tokens/day - limited)
        """
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        
    def create_prompt(self, question, data_source):
        """
        Create a prompt for SQL generation with assumptions and confidence
        
        Args:
            question (str): Business question
            data_source (str): Data source (sales_dw or marketing_dw)
            
        Returns:
            str: Formatted prompt
        """
        schema = SALES_DW_SCHEMA if data_source == "sales_dw" else MARKETING_DW_SCHEMA
        
        # Format the prompt template with schema and question
        prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)
        
        return prompt
    
    def generate_query(self, question, data_source):
        """
        Generate SQL query using Groq API with assumptions and confidence
        
        Args:
            question (str): Business question
            data_source (str): Data source (sales_dw or marketing_dw)
            
        Returns:
            dict: Dictionary with 'sql', 'assumptions', and 'confidence' keys
        """
        prompt = self.create_prompt(question, data_source)
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SQL developer. Generate SQL queries with assumptions and confidence scores. Respond with valid JSON containing 'sql', 'assumptions', and 'confidence' fields."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.1,  # Low temperature for consistent, deterministic output
                max_tokens=2048,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"}  # Enforce JSON output
            )
            
            response = chat_completion.choices[0].message.content
            
            logger.debug(json.dumps(response, indent=2))
            
            # Parse JSON response (guaranteed valid JSON due to response_format)
            result = json.loads(response)
            
            # Extract required fields with defaults
            sql = result.get('sql', '')
            assumptions = result.get('assumptions', 'No assumptions documented')
            confidence = result.get('confidence', 'MEDIUM')
            
            return {
                'sql': sql,
                'assumptions': assumptions,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'sql': f"-- Error generating query: {str(e)}",
                'assumptions': f"Error occurred: {str(e)}",
                'confidence': "LOW"
            }
    
    def generate_queries_batch(self, questions_list, rpm=5):
        """
        Generate SQL queries for a batch of questions with rate limiting for free tier.
        
        Args:
            questions_list (list): List of question dictionaries with 'question' and 'data_source' keys
            rpm (int): Requests per minute (1-20 recommended for free tier)
                For llama-3.1-8b-instant free tier: max 30 RPM, but 10 RPM recommended
                to stay safely under token per minute (TPM) limits as well
            
        Returns:
            list: List of dictionaries with question info, query, assumptions, and confidence
        
        Raises:
            ValueError: If rpm is not between 1 and 10
        """
        # Validate rpm parameter
        if not isinstance(rpm, int) or rpm < 1 or rpm > 10:
            raise ValueError("requests per minute must be an integer between 1 and 10")
        
        # Calculate delay between requests to achieve target RPM
        delay_seconds = 60 / rpm
        
        results = []
        total_questions = len(questions_list)
        
        logger.info(f"Processing {total_questions} questions at {rpm} requests per minute")
        logger.info(f"Delay between requests: {delay_seconds:.2f}s")
        logger.info(f"Estimated completion time: ~{(total_questions / rpm):.1f} minutes")
        
        for idx, q in enumerate(questions_list, 1):
            logger.info(f"[{idx}/{total_questions}] Generating query for Question {q['id']}: {q['question'][:60]}...")
            
            query_result = self.generate_query(q['question'], q['data_source'])
            
            results.append({
                'question_id': q['id'],
                'question': q['question'],
                'target_source': q['data_source'],
                'sql': query_result['sql'],
                'assumptions': query_result['assumptions'],
                'confidence': query_result['confidence']
            })
            
            # Add delay between requests to maintain target RPM (except for last request)
            if idx < total_questions:
                logger.debug(f"Rate limiting: waiting {delay_seconds:.2f}s (RPM: {rpm})...")
                time.sleep(delay_seconds)
            
        return results
