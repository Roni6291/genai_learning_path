# GenAI Learning Path - SQL Generation & Text-to-Image

This repository contains two projects for the GenAI learning path:
1. **SQL Query Generator** - Generates SQL queries using Groq API
2. **Text-to-Image Generator** - Creates images using Hugging Face Stable Diffusion

---

## Project 1: SQL Query Generator for DWBI Team

Generates SQL queries for business intelligence questions using Groq API with free tier optimized models.

### Overview

As part of the Data Warehouse & Business Intelligence (DWBI) team at a retail company, this tool helps automatically generate SQL queries for ad-hoc business questions across two data sources:
- **Sales Data Warehouse (sales_dw)** - Sales transactions and product information
- **Marketing Data Warehouse (marketing_dw)** - Campaign performance and impression metrics

### Features

- ✅ Generates SQL queries for 20 business questions
- ✅ Uses Groq API with **llama-3.1-8b-instant** (free tier optimized)
- ✅ **JSON Mode** for guaranteed structured responses
- ✅ Rate limiting (5 RPM default) for free tier compliance
- ✅ Outputs to CSV: `dwbi_sql_output.csv`
- ✅ Includes assumptions and confidence scores for each query
- ✅ **Loguru logging** for professional output
- ✅ **External configuration files** (schemas, prompts, questions)
- ✅ Comprehensive error handling

### Project Structure

```
1.5-sql_generation/
├── main.py                           # Main Python script
├── sql_generator.py                  # SQL query generator using Groq API
├── questions.json                    # 20 business questions (JSON format)
├── table_schemas/
│   ├── sales_dw_schema.txt          # Sales Data Warehouse schema
│   └── marketing_dw_schema.txt      # Marketing Data Warehouse schema
├── prompt_template.txt               # LLM prompt template
├── pyproject.toml                    # Dependencies (uv package manager)
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore file
├── README.md                         # This file
└── dwbi_sql_output.csv              # Generated output (after running)
```

### Setup Instructions

#### 1. Prerequisites

- Python 3.10 or higher
- Groq API key (get one at https://console.groq.com/)
- uv package manager (recommended) or pip

#### 2. Installation

```bash
# Navigate to the project directory
cd 1.5-sql_generation

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

#### 3. Configuration

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

#### 4. Run the SQL Generator

```bash
python main.py
```

The script will:
- Generate SQL queries for all 20 business questions
- Apply rate limiting (5 requests per minute by default)
- Save results to `dwbi_sql_output.csv`
- Display progress with loguru logging

## Database Schema

### Sales Data Warehouse (sales_dw)

**Table: sales**
| Column | Type | Description |
|--------|------|-------------|
| sale_id | INT | Unique identifier for each sale |
| product_id | INT | Foreign key → products.product_id |
| region | VARCHAR | Sales region |
| sale_date | DATE | Date of transaction |
| sales_amount | DECIMAL | Revenue from transaction |
| quantity | INT | Number of units sold |

**Table: products**
| Column | Type | Description |
|--------|------|-------------|
| product_id | INT | Unique product ID |
| product_name | VARCHAR | Name of the product |
| category | VARCHAR | Product category |
| subcategory | VARCHAR | Subcategory |
| brand | VARCHAR | Product brand |

### Marketing Data Warehouse (marketing_dw)

**Table: campaigns**
| Column | Type | Description |
|--------|------|-------------|
| campaign_id | INT | Unique campaign ID |
| channel | VARCHAR | Marketing channel |
| start_date | DATE | Campaign start date |
| end_date | DATE | Campaign end date |
| budget | DECIMAL | Campaign budget |

**Table: impressions**
| Column | Type | Description |
|--------|------|-------------|
| campaign_id | INT | Foreign key → campaigns.campaign_id |
| day | DATE | Date of impressions |
| impressions | INT | Number of impressions shown |
| clicks | INT | Number of clicks received |

**Relationships:** impressions.campaign_id → campaigns.campaign_id

**Key Metrics:**
- CTR (Click-Through Rate) = (clicks / impressions) × 100
- CPC (Cost Per Click) = budget / total_clicks
- Conversion Ratio = clicks / impressions

## Business Questions Coverage

The tool generates SQL queries for 20 business questions:

### Sales Questions (1-10)
1. Top 5 products by sales amount (last 90 days)
2. Month-over-month sales growth by region (past 6 months)
3. Categories contributing most to revenue (last year)
4. Average order value per region (current quarter)
5. Top 3 brands by quantity sold (last 30 days)
6. Subcategory with sharpest sales decline (vs previous quarter)
7. Regional percentage contribution to total sales (this year)
8. Sales amount vs quantity trend for Electronics
9. Product with highest sales per unit (last 60 days)
10. Top 10 customers by revenue

### Marketing Questions (11-20)
11. Channel with highest impressions (last quarter)
12. Average CTR per channel (last month)
13. Campaign with lowest CPC (last 6 months)
14. Total budget spent per channel (last year)
15. Top 3 campaigns by impressions
16. Daily average impressions vs clicks for Social Media
17. Channel with highest conversion ratio
18. Campaigns running >60 days with total spend
19. Underperforming campaigns (budget vs clicks)
20. Month with highest total impressions

## Output Format

After running the script, you'll get a CSV file named **`dwbi_sql_output.csv`** with exactly 20 rows (one per business question).

### CSV Columns:
| Column | Description |
|--------|-------------|
| **question_id** | Question number (1-20) |
| **question** | The business question text |
| **target_source** | Data source (sales_dw or marketing_dw) |
| **sql** | Generated SQL query |
| **assumptions** | Any assumptions made during query generation |
| **confidence** | Confidence level (HIGH, MEDIUM, or LOW) |

### Confidence Levels:
- **HIGH**: Query can be fully satisfied with available schema
- **MEDIUM**: Query requires reasonable assumptions or aggregations
- **LOW**: Schema may not have all required data or question is ambiguous

### Model Configuration

The project uses **llama-3.1-8b-instant** by default (free tier optimized). You can change the model in `main.py`:

**Free Tier Models:**
- `llama-3.1-8b-instant` (default) - 30 RPM, 6K TPM, 14.4K req/day
- `llama-3.3-70b-versatile` - 30 RPM, 6K TPM, 1K req/day (limited)
- `mixtral-8x7b-32768` - 30 RPM, 5K TPM

To change the model, edit line in `main.py`:
```python
generator = SQLQueryGenerator(model_name="llama-3.1-8b-instant")
```

**Rate Limiting:**
Adjust RPM in `main.py` (1-10 recommended for free tier):
```python
results = generator.generate_queries_batch(business_questions, rpm=5)
```

## Example Usage

### Python Script Example
```python
from sql_generator import SQLQueryGenerator
import os

# Load API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize generator
generator = SQLQueryGenerator(model_name="llama-3.3-70b-versatile")

# Generate a single query
question = "What are the top 5 products by sales amount in the last 90 days?"
result = generator.generate_query(question, data_source="sales_dw")

print(f"SQL: {result['sql']}")
print(f"Assumptions: {result['assumptions']}")
print(f"Confidence: {result['confidence']}")

# Generate queries for all questions
from questions import BUSINESS_QUESTIONS
results = generator.generate_queries_batch(BUSINESS_QUESTIONS)
```

### Sample Output
```
Question 1: What are the top 5 products by sales amount in the last 90 days?
SQL: SELECT p.product_name, SUM(s.sales_amount) as total_sales...
Assumptions: Used CURRENT_DATE for date calculations, assumed 90 days means last 3 months
Confidence: HIGH
```

## Troubleshooting

### Error: GROQ_API_KEY not found
- Make sure you've created a `.env` file (copy from `.env.example`)
- Add your Groq API key to the `.env` file: `GROQ_API_KEY=your_key_here`
- Ensure there are no spaces around the `=` sign
- For Jupyter Notebook, you can also set it directly: `os.environ["GROQ_API_KEY"] = "your_key"`

### API Rate Limits
### Error: GROQ_API_KEY not found
- Make sure you've created a `.env` file (copy from `.env.example`)
- Add your Groq API key to the `.env` file: `GROQ_API_KEY=your_key_here`
- Ensure there are no spaces around the `=` sign
- The script handles JSON parsing failures gracefully
- If a query fails, check the `assumptions` column for error details
- The `confidence` will be set to LOW for failed queries

### Model Selection
- **llama-3.3-70b-versatile** (default, recommended) - Best for complex SQL queries
- **llama-3.1-70b-versatile** - Good alternative
- **mixtral-8x7b-32768** - Faster, good for simpler queries
- **gemma2-9b-it** - Fastest, but may be less accurate for complex queries

## License

This project is for educational purposes as part of the GenAI learning path.

## How It Works

1. **Schema Loading**: Loads database schemas for both Sales and Marketing data warehouses
2. **Prompt Engineering**: Creates specialized prompts with schema context and instructions
3. **API Call**: Sends requests to Groq API using LLaMA 3.3 70B model
4. **Response Parsing**: Extracts SQL query, assumptions, and confidence from JSON response
5. **Validation**: Validates response format and handles parsing errors
6. **CSV Output**: Writes results to `dwbi_sql_output.csv` with all required columns

## Advanced Features

### Custom API Configuration
You can use different LLM providers by modifying the generator:
- **Groq API** (default): Fast inference with LLaMA models
- **OpenAI API**: Replace Groq client with OpenAI client
- **Ollama**: For local model inference

### Extending Questions
Add more questions in `questions.py`:
```python
{
    "id": 21,
    "data_source": "sales_dw",
    "question": "Your custom business question here"
}
```

## Contributing

This is a learning project. Feel free to experiment with:
- Different Groq models or LLM providers
- Additional business questions (extend beyond 20)
- Query optimization techniques
- Enhanced prompt engineering
- Output format enhancements (JSON, Excel, etc.)

## Contact

For questions about this project, please refer to the GenAI learning path documentation.
