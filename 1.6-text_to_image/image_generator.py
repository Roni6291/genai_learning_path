"""
Text-to-Image Generator using Hugging Face Inference API
Generates images using Stable Diffusion v1.5 model
"""

import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from huggingface_hub import InferenceClient


class TextToImageGenerator:
    """
    Generates images from text prompts using Hugging Face Inference API
    """
    
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell"):
        """
        Initialize the Text-to-Image Generator
        
        Args:
            model_name (str): Hugging Face model to use for image generation
                Default: black-forest-labs/FLUX.1-schnell
        """
        load_dotenv()
        self.api_key = os.getenv("HF_API_KEY")
        
        if not self.api_key:
            raise ValueError("HF_API_KEY not found in environment variables. Please set it in .env file")
        
        self.model_name = model_name
        self.client = InferenceClient(api_key=self.api_key)
        
        logger.info(f"Initialized Text-to-Image Generator with model: {model_name}")
        logger.info("Using Hugging Face Inference API (online)")
    
    def generate_image(self, prompt, output_path="output/dashboard.png"):
        """
        Generate an image from a text prompt
        
        Args:
            prompt (str): Text description of the image to generate
            output_path (str): Path where the generated image will be saved
            
        Returns:
            dict: Dictionary with 'success', 'output_path', 'model_name', and 'prompt' keys
        """
        logger.info(f"Generating image with prompt: {prompt[:100]}...")
        
        try:
            # Generate image using InferenceClient
            image = self.client.text_to_image(prompt, model=self.model_name)
            
            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            image.save(output_file)
            logger.success(f"Image saved successfully to: {output_file.absolute()}")
            
            # Log image details
            logger.info(f"Image size: {image.size}")
            logger.info(f"Image mode: {image.mode}")
            
            return {
                'success': True,
                'output_path': str(output_file.absolute()),
                'model_name': self.model_name,
                'prompt': prompt,
                'image_size': image.size,
                'image_mode': image.mode
            }
            
        except Exception as e:
            error_msg = f"Error generating image: {type(e).__name__}: {str(e)}"
            logger.exception("Full error details:")
            return {
                'success': False,
                'error': error_msg,
                'model_name': self.model_name,
                'prompt': prompt
            }


def main():
    """
    Main function to generate dashboard visualization
    """
    logger.info("=" * 80)
    logger.info("Text-to-Image Generator")
    logger.info("Using Hugging Face Stable Diffusion v1.5")
    logger.info("=" * 80)
    
    # Define the prompt
    prompt = (
        "Visualize the sales performance dashboard of the retail company, "
        "showing products, regions, and campaigns as icons or infographics."
    )
    
    # Initialize generator
    try:
        generator = TextToImageGenerator(model_name="black-forest-labs/FLUX.1-schnell")
    except ValueError as e:
        logger.error(str(e))
        logger.info("Please create a .env file with your HF_API_KEY")
        return
    
    # Generate image
    logger.info("Starting image generation...")
    logger.info("-" * 80)
    
    result = generator.generate_image(
        prompt=prompt,
        output_path="output/dashboard.png"
    )
    
    # Display results
    logger.info("-" * 80)
    if result['success']:
        logger.success("Image generation completed successfully!")
        logger.info(f"Model used: {result['model_name']}")
        logger.info(f"Prompt: {result['prompt']}")
        logger.info(f"Output path: {result['output_path']}")
        logger.info(f"Image size: {result['image_size']}")
        
        # Create documentation
        doc_path = Path("output/generation_info.txt")
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("Text-to-Image Generation Documentation\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model Name: {result['model_name']}\n")
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Output Path: {result['output_path']}\n")
            f.write(f"Image Size: {result['image_size']}\n")
            f.write(f"Image Mode: {result['image_mode']}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.success(f"Documentation saved to: {doc_path.absolute()}")
    else:
        logger.error("Image generation failed!")
        logger.error(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
