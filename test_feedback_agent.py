#!/usr/bin/env python3
"""
Test script for the Feedback Agent functionality
This script demonstrates how the feedback agent works without the full Streamlit UI
"""

import os
import sys
from PIL import Image, ImageEnhance, ImageFilter
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeedbackAgentTester:
    def __init__(self):
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def apply_image_adjustments(self, image_path: str, feedback: str) -> str:
        """Apply direct image adjustments based on feedback"""
        try:
            with Image.open(image_path) as image:
                feedback_lower = feedback.lower()
                adjusted_image = image.copy()
                
                # Brightness adjustments
                if 'brighter' in feedback_lower or 'bright' in feedback_lower:
                    enhancer = ImageEnhance.Brightness(adjusted_image)
                    adjusted_image = enhancer.enhance(1.3)
                    logger.info("Applied brightness enhancement")
                    
                elif 'darker' in feedback_lower or 'dark' in feedback_lower:
                    enhancer = ImageEnhance.Brightness(adjusted_image)
                    adjusted_image = enhancer.enhance(0.7)
                    logger.info("Applied darkness enhancement")
                    
                # Color adjustments
                if 'colorful' in feedback_lower or 'more color' in feedback_lower:
                    enhancer = ImageEnhance.Color(adjusted_image)
                    adjusted_image = enhancer.enhance(1.4)
                    logger.info("Applied color enhancement")
                    
                elif 'less color' in feedback_lower or 'desaturate' in feedback_lower:
                    enhancer = ImageEnhance.Color(adjusted_image)
                    adjusted_image = enhancer.enhance(0.6)
                    logger.info("Applied color desaturation")
                    
                # Contrast adjustments
                if 'more contrast' in feedback_lower or 'sharper' in feedback_lower:
                    enhancer = ImageEnhance.Contrast(adjusted_image)
                    adjusted_image = enhancer.enhance(1.3)
                    logger.info("Applied contrast enhancement")
                    
                elif 'softer' in feedback_lower or 'less contrast' in feedback_lower:
                    enhancer = ImageEnhance.Contrast(adjusted_image)
                    adjusted_image = enhancer.enhance(0.8)
                    logger.info("Applied contrast reduction")
                    
                # Filter effects
                if 'blur' in feedback_lower or 'soft' in feedback_lower:
                    adjusted_image = adjusted_image.filter(ImageFilter.GaussianBlur(radius=1))
                    logger.info("Applied blur filter")
                    
                elif 'sharp' in feedback_lower or 'crisp' in feedback_lower:
                    adjusted_image = adjusted_image.filter(ImageFilter.SHARPEN)
                    logger.info("Applied sharpen filter")
                
                # Save the adjusted image
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                adjusted_path = os.path.join(self.output_dir, f"{base_name}_adjusted.png")
                adjusted_image.save(adjusted_path, format='PNG')
                
                logger.info(f"Saved adjusted image: {adjusted_path}")
                return adjusted_path
                
        except Exception as e:
            logger.error(f"Image adjustment failed: {e}")
            return None
    
    def test_feedback_options(self, image_path: str):
        """Test various feedback options on an image"""
        if not os.path.exists(image_path):
            logger.error(f"Test image not found: {image_path}")
            return
        
        feedback_tests = [
            "make it brighter",
            "make it darker", 
            "more colorful",
            "less color",
            "more contrast",
            "softer",
            "blur",
            "sharper"
        ]
        
        logger.info(f"Testing feedback agent with image: {image_path}")
        
        for feedback in feedback_tests:
            logger.info(f"Testing feedback: '{feedback}'")
            result_path = self.apply_image_adjustments(image_path, feedback)
            if result_path:
                logger.info(f"✅ Successfully applied: {feedback} -> {result_path}")
            else:
                logger.error(f"❌ Failed to apply: {feedback}")
        
        logger.info("Feedback agent testing completed!")

def main():
    """Main function to run the feedback agent test"""
    tester = FeedbackAgentTester()
    
    # Look for test images in the output directory
    output_dir = "output"
    test_images = []
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'generated' in file:
                test_images.append(os.path.join(output_dir, file))
    
    if not test_images:
        logger.warning("No generated images found for testing. Please generate an image first using the main app.")
        logger.info("You can run: streamlit run app.py")
        return
    
    # Use the most recent generated image
    test_image = test_images[-1]
    logger.info(f"Using test image: {test_image}")
    
    # Run the feedback tests
    tester.test_feedback_options(test_image)
    
    print("\n" + "="*50)
    print("FEEDBACK AGENT TEST COMPLETED")
    print("="*50)
    print(f"Original image: {test_image}")
    print(f"Adjusted images saved in: {output_dir}")
    print("\nTest different feedback phrases:")
    print("- 'make it brighter' / 'make it darker'")
    print("- 'more colorful' / 'less color'") 
    print("- 'more contrast' / 'softer'")
    print("- 'blur' / 'sharper'")

if __name__ == "__main__":
    main()
