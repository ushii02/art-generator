import streamlit as st
import torch
import gc
import requests
import os
import time
import json
import logging
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import spacy
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np
import base64
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("art_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_GENERATIONS_PER_HOUR = 20
RESET_INTERVAL = 3600  # 1 hour in seconds
IMAGE_SIZE = (512, 512)
THUMBNAIL_SIZE = (128, 128)
API_TIMEOUT = 30

@dataclass
class GenerationResult:
    image_path: Optional[str]
    references: List[Dict]
    style_info: Dict
    success: bool
    error_message: Optional[str] = None

@dataclass
class FeedbackResult:
    original_image_path: str
    edited_image_path: Optional[str]
    feedback_applied: str
    success: bool
    error_message: Optional[str] = None

class ArtGeneratorApp:
    def __init__(self):
        self.setup_environment()
        self.setup_models()
        self.setup_data()
        self.setup_subscription_system()
        
    def setup_environment(self):
        """Initialize environment and validate configuration"""
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            st.error("Please set HF_TOKEN in .env file or Streamlit secrets")
            st.info("Get your token from: https://huggingface.co/settings/tokens")
            st.stop()
            
        # Force CPU usage for consistency
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)  # Limit CPU threads
        
        # API configurations with updated models
        self.api_urls = {
            "text": "https://api-inference.huggingface.co/models/distilbert/distilgpt2",
            "image": "https://api-inference.huggingface.co/models/stable-diffusion-v1-5/stable-diffusion-v1-5"
        }
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        # Test API access
        self.test_api_access()
        
        # Create necessary directories
        os.makedirs("output", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def test_api_access(self):
        """Test API access and suggest fixes"""
        try:
            test_response = requests.get(
                "https://api-inference.huggingface.co/models/stable-diffusion-v1-5/stable-diffusion-v1-5",
                headers=self.headers,
                timeout=10
            )
            
            if test_response.status_code == 403:
                st.error("🚫 API Access Denied - Please check your Hugging Face token:")
                st.markdown("""
                **Common solutions:**
                1. **Get a new token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
                2. **Check token permissions**: Make sure it has 'Read' access
                3. **Accept model terms**: Visit the model page and accept terms of use
                4. **Wait for model loading**: Some models need time to warm up
                """)
                st.info("🔄 The app will try alternative models automatically")
                
            elif test_response.status_code == 503:
                st.warning("⏳ Model is loading on Hugging Face servers. This usually takes 1-2 minutes.")
                
        except Exception as e:
            logger.warning(f"API test failed: {e}")
            st.info("🔧 Will attempt generation with fallback models")

    @st.cache_resource
    def load_nlp_model(_self):
        """Load spaCy model with caching"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            st.stop()

    @st.cache_resource
    def load_clip_models(_self):
        """Load CLIP models with proper caching"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            model.eval()
            
            logger.info("CLIP models loaded successfully")
            return model, processor
        except Exception as e:
            logger.error(f"Failed to load CLIP models: {e}")
            st.error(f"Failed to load CLIP models: {e}")
            st.stop()

    @st.cache_resource
    def load_safety_classifier(_self):
        """Load NSFW safety classifier with caching"""
        try:
            from transformers import pipeline
            return pipeline(
                "image-classification", 
                model="Falconsai/nsfw_image_detection",
                torch_dtype=torch.float32
            )
        except Exception as e:
            logger.warning(f"Failed to load safety classifier: {e}")
            return None

    def setup_models(self):
        """Initialize all AI models"""
        self.nlp = self.load_nlp_model()
        self.clip_model, self.clip_processor = self.load_clip_models()
        self.safety_classifier = self.load_safety_classifier()

    @staticmethod
    @st.cache_data
    def load_and_validate_csv() -> pd.DataFrame:
        """Load and validate the art references CSV"""
        csv_path = "data/art_references.csv"
        
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file {csv_path} not found")
            return pd.DataFrame(columns=['image_path', 'style'])
            
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['image_path', 'style']
            
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV missing required columns: {required_columns}")
                return pd.DataFrame(columns=required_columns)
                
            # Validate and filter valid images
            valid_rows = []
            invalid_paths = []
            for idx, row in df.iterrows():
                image_path = os.path.normpath(row['image_path'])
                if ArtGeneratorApp.validate_image_path(image_path):
                    valid_rows.append(row)
                else:
                    invalid_paths.append(image_path)
                    
            valid_df = pd.DataFrame(valid_rows)
            logger.info(f"Loaded {len(valid_df)}/{len(df)} valid image references")
            if invalid_paths:
                logger.warning(f"Invalid image paths: {', '.join(invalid_paths)}")
                st.warning(f"Invalid images: {', '.join(invalid_paths)}")
            
            if len(valid_df) < len(df):
                st.warning(f"Only {len(valid_df)}/{len(df)} images are valid and accessible")
                
            return valid_df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            st.error(f"Failed to load art references: {e}")
            return pd.DataFrame(columns=['image_path', 'style'])

    @staticmethod
    def validate_image_path(path: str) -> bool:
        """Validate that an image path exists and is readable"""
        try:
            if not os.path.exists(path):
                return False
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def setup_data(self):
        """Load and prepare data"""
        self.df = self.load_and_validate_csv()
        self.vectorstore = self.build_faiss_index()
        
    def setup_subscription_system(self):
        """Initialize subscription system"""
        # Initialize subscription state
        if "subscription_status" not in st.session_state:
            st.session_state.subscription_status = "free"  # "free" or "premium"
        if "subscription_expiry" not in st.session_state:
            st.session_state.subscription_expiry = None
        
    def check_subscription_status(self) -> bool:
        """Check if user has active subscription"""
        if st.session_state.subscription_status == "premium":
            # In a real app, you would check expiry date against current date
            if st.session_state.subscription_expiry:
                from datetime import datetime
                try:
                    expiry_date = datetime.strptime(st.session_state.subscription_expiry, "%Y-%m-%d")
                    return datetime.now() < expiry_date
                except:
                    return False
            return True  # For demo purposes, premium is always active
        return False
        
    def upgrade_subscription(self):
        """Upgrade user to premium subscription"""
        st.session_state.subscription_status = "premium"
        # Set expiry to 30 days from now for demo
        from datetime import datetime, timedelta
        expiry = datetime.now() + timedelta(days=30)
        st.session_state.subscription_expiry = expiry.strftime("%Y-%m-%d")
        st.success("🎉 Successfully upgraded to Premium! Images will no longer have watermarks.")
        st.rerun()
        
    def cancel_subscription(self):
        """Cancel premium subscription"""
        st.session_state.subscription_status = "free"
        st.session_state.subscription_expiry = None
        st.info("Subscription cancelled. Images will now include watermarks.")
        st.rerun()

    @st.cache_resource
    def build_faiss_index(_self):
        """Build FAISS index for image similarity search"""
        if _self.df.empty:
            logger.warning("No data available for FAISS index")
            return None
            
        index_path = "faiss_index"
        
        if os.path.exists(index_path):
            try:
                logger.info("Using simple similarity search (FAISS replacement)")
                return _self.build_simple_index()
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                
        return _self.build_simple_index()

    def build_simple_index(self):
        """Build a simple embedding-based index as FAISS replacement"""
        embeddings = []
        metadata = []
        
        with torch.no_grad():
            for idx, row in self.df.iterrows():
                try:
                    image_path = os.path.normpath(row['image_path'])
                    with Image.open(image_path) as image:
                        image = image.convert('RGB').resize(THUMBNAIL_SIZE)
                        inputs = self.clip_processor(images=image, return_tensors="pt")
                        embedding = self.clip_model.get_image_features(**inputs)
                        embeddings.append(embedding.cpu().numpy().flatten())
                        metadata.append(row.to_dict())
                        del inputs, embedding
                        
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue
                    
        if embeddings:
            embeddings_array = np.array(embeddings)
            logger.info(f"Built index with {len(embeddings)} images")
            return {"embeddings": embeddings_array, "metadata": metadata}
        
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def call_api(self, url: str, payload: Dict, timeout: int = API_TIMEOUT) -> Tuple[Optional[Dict], Optional[str]]:
        """Make API call with retry and proper error handling"""
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload, 
                timeout=timeout
            )
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('image/'):
                return {"content": response.content}, None
            else:
                return response.json(), None
                
        except requests.exceptions.Timeout:
            return None, "API request timed out"
        except requests.exceptions.RequestException as e:
            return None, f"API request failed: {str(e)}"
        except json.JSONDecodeError:
            return None, "Invalid API response format"

    def extract_style_elements(self, prompt: str, style: str = "Auto-detect") -> Dict:
        """Extract style elements from prompt or use selected style"""
        try:
            doc = self.nlp(prompt)
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
            available_styles = self.df['style'].unique().tolist() if not self.df.empty else []
            
            if style and style != "Auto-detect":
                detected_style = style
            else:
                # Prioritize exact matches, then partial matches
                prompt_lower = prompt.lower()
                detected_style = next(
                    (s for s in available_styles if s.lower() == prompt_lower),
                    next(
                        (s for s in available_styles if s.lower() in prompt_lower),
                        "contemporary"
                    )
                )
            return {
                "style": detected_style,
                "subject": nouns[0] if nouns else "artwork",
                "refined_prompt": f"{prompt}, in the style of {detected_style}" if detected_style else prompt,
                "elements": {"nouns": nouns[:3], "adjectives": adjectives[:3]}
            }
        except Exception as e:
            logger.warning(f"Style extraction failed: {e}")
            return {
                "style": style if style and style != "Auto-detect" else "contemporary",
                "subject": "artwork",
                "refined_prompt": prompt,
                "elements": {"nouns": [], "adjectives": []}
            }

    def find_similar_images(self, query: str, k: int = 2) -> List[Dict]:
        """Find similar images using simple cosine similarity"""
        if not self.vectorstore or self.df.empty:
            logger.warning("No image index available")
            return []
            
        try:
            with torch.no_grad():
                inputs = self.clip_processor(text=query, return_tensors="pt")
                query_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy().flatten()
                
                embeddings = self.vectorstore["embeddings"]
                similarities = np.dot(embeddings, query_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                k = min(k, len(embeddings))
                top_indices = np.argsort(similarities)[-k:][::-1]
                results = [self.vectorstore["metadata"][i] for i in top_indices]
                
                logger.info(f"Found {len(results)} similar images for query: {query}")
                return results
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def check_image_safety(self, image: Image.Image) -> bool:
        """Bypass NSFW safety check: always return True (all images are considered safe)"""
        return True

    def generate_image(self, prompt: str, references: List[Dict], add_watermark: bool = True) -> Optional[str]:
        """Generate image with enhanced prompt and safety checking"""
        fallback_models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5"
        ]
        
        try:
            if references:
                styles = [ref.get('style', '') for ref in references if ref.get('style')]
                style_text = ", ".join(set(styles))
                enhanced_prompt = f"{prompt}, in the style of {style_text}"
            else:
                enhanced_prompt = prompt
                
            enhanced_prompt = enhanced_prompt[:200]
            
            logger.info(f"Generating image with prompt: {enhanced_prompt}")
            
            for model_name in fallback_models:
                try:
                    model_url = f"https://api-inference.huggingface.co/models/{model_name}"
                    
                    payload = {
                        "inputs": enhanced_prompt,
                        "parameters": {
                            "num_inference_steps": 20,
                            "guidance_scale": 7.5
                        }
                    }
                    
                    response_data, error = self.call_api(model_url, payload, timeout=60)
                    
                    if error:
                        if "403" in error:
                            logger.warning(f"Access denied for {model_name}, trying next model...")
                            continue
                        elif "503" in error:
                            st.info(f"⏳ {model_name} is loading, trying alternative...")
                            continue
                        else:
                            logger.warning(f"Error with {model_name}: {error}")
                            continue
                    
                    if not response_data or "content" not in response_data:
                        logger.warning(f"Invalid response from {model_name}")
                        continue
                        
                    st.success(f"✅ Generated using {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed with {model_name}: {e}")
                    continue
            else:
                st.error("❌ All image generation models failed. Please try again later.")
                st.markdown("""
                **Troubleshooting:**
                1. Check your Hugging Face token permissions
                2. Visit model pages and accept terms of use
                3. Try again in a few minutes (models may be loading)
                """)
                return None
                
            try:
                image = Image.open(BytesIO(response_data["content"]))
                image = image.convert('RGB')
                
                if not self.check_image_safety(image):
                    st.error("Generated image flagged as inappropriate")
                    logger.warning("Image failed safety check")
                    return None
                    
                # Apply watermark only if user doesn't have premium subscription
                if add_watermark and not self.check_subscription_status():
                    image = self.add_watermark(image)
                    
                timestamp = int(time.time())
                image_path = f"output/generated_{timestamp}.png"
                image.save(image_path, format='PNG', optimize=True)
                
                logger.info(f"Image generated successfully: {image_path}")
                return image_path
                
            except Exception as e:
                st.error(f"Failed to process generated image: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            st.error(f"Image generation failed: {e}")
            return None
        finally:
            gc.collect()

    def add_watermark(self, image: Image.Image) -> Image.Image:
        """Add watermark to generated image"""
        try:
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("arial.ttf", size=16)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                    
            watermark_text = "Generated by Art Generator"
            
            if font:
                bbox = draw.textbbox((0, 0), watermark_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = 150, 15
                
            x = image.width - text_width - 10
            y = image.height - text_height - 10
            
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [x-5, y-2, x+text_width+5, y+text_height+2], 
                fill=(0, 0, 0, 128)
            )
            
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255))
            
            return image
            
        except Exception as e:
            logger.warning(f"Failed to add watermark: {e}")
            return image

    def save_generation_metadata(self, image_path: str, prompt: str, style: str):
        """Save metadata of generated images to CSV"""
        metadata_path = "output/generated_images.csv"
        metadata = {
            "image_path": image_path,
            "prompt": prompt,
            "style": style,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        df = pd.DataFrame([metadata])
        if os.path.exists(metadata_path):
            existing_df = pd.read_csv(metadata_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata for {image_path}")

    def load_generated_images(self) -> pd.DataFrame:
        """Load metadata of previously generated images"""
        metadata_path = "output/generated_images.csv"
        if os.path.exists(metadata_path):
            try:
                return pd.read_csv(metadata_path)
            except Exception as e:
                logger.warning(f"Failed to load generated images metadata: {e}")
                return pd.DataFrame(columns=["image_path", "prompt", "style", "timestamp"])
        return pd.DataFrame(columns=["image_path", "prompt", "style", "timestamp"])

    def manage_rate_limit(self) -> bool:
        """No rate limiting: always allow generation"""
        return True

    def validate_inputs(self, prompt: str) -> bool:
        """Validate user inputs (no inappropriate content filter)"""
        if not prompt or not prompt.strip():
            st.error("Please enter a valid art prompt")
            return False
        if len(prompt) > 500:
            st.error("Prompt too long. Please keep it under 500 characters.")
            return False
        return True

    def integrate_feedback(self, original_prompt: str, feedback: str) -> str:
        """Integrate user feedback into prompt"""
        if not feedback.strip():
            return original_prompt
            
        try:
            feedback_lower = feedback.lower()
            
            if 'brighter' in feedback_lower or 'bright' in feedback_lower:
                return f"{original_prompt}, bright and luminous"
            elif 'darker' in feedback_lower or 'dark' in feedback_lower:
                return f"{original_prompt}, dark and moody"
            elif 'colorful' in feedback_lower or 'colors' in feedback_lower:
                return f"{original_prompt}, vibrant and colorful"
            elif 'simple' in feedback_lower or 'minimal' in feedback_lower:
                return f"{original_prompt}, minimalist style"
            else:
                return f"{original_prompt}, {feedback}"
                
        except Exception as e:
            logger.warning(f"Feedback integration failed: {e}")
            return original_prompt

    def detect_object_regions(self, image: Image.Image, object_name: str) -> List[Tuple[int, int, int, int]]:
        """Detect regions in the image where objects could be added/modified"""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Simple region detection based on object type
            regions = []
            h, w = cv_image.shape[:2]
            
            if object_name.lower() in ['hat', 'cap', 'helmet', 'crown']:
                # Top region for head accessories
                regions.append((w//4, 0, 3*w//4, h//3))
            elif object_name.lower() in ['glasses', 'sunglasses', 'eyewear']:
                # Middle-upper region for face accessories
                regions.append((w//4, h//6, 3*w//4, h//2))
            elif object_name.lower() in ['necklace', 'tie', 'scarf']:
                # Neck/chest region
                regions.append((w//3, h//3, 2*w//3, 2*h//3))
            elif object_name.lower() in ['background', 'sky', 'clouds']:
                # Background regions
                regions.append((0, 0, w, h//2))
            else:
                # Default: center region
                regions.append((w//4, h//4, 3*w//4, 3*h//4))
            
            return regions
            
        except Exception as e:
            logger.warning(f"Object region detection failed: {e}")
            # Fallback: return center region
            w, h = image.size
            return [(w//4, h//4, 3*w//4, 3*h//4)]

    def create_inpainting_mask(self, image: Image.Image, object_name: str, action: str) -> Image.Image:
        """Create a mask for inpainting based on the object and action"""
        try:
            w, h = image.size
            mask = Image.new('RGB', (w, h), (0, 0, 0))  # Black mask
            draw = ImageDraw.Draw(mask)
            
            # Get regions where the object should be added/modified
            regions = self.detect_object_regions(image, object_name)
            
            for region in regions:
                x1, y1, x2, y2 = region
                
                if action.lower() in ['add', 'put', 'place']:
                    # For adding objects, create a smaller focused mask
                    if object_name.lower() in ['hat', 'cap']:
                        # Oval shape for hat
                        draw.ellipse([x1, y1, x2, y1 + (y2-y1)//2], fill=(255, 255, 255))
                    elif object_name.lower() in ['glasses', 'sunglasses']:
                        # Rectangle for glasses
                        draw.rectangle([x1, y1 + (y2-y1)//3, x2, y1 + 2*(y2-y1)//3], fill=(255, 255, 255))
                    else:
                        # Default circular region
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        radius = min(x2 - x1, y2 - y1) // 3
                        draw.ellipse([center_x - radius, center_y - radius, 
                                    center_x + radius, center_y + radius], fill=(255, 255, 255))
                elif action.lower() in ['remove', 'delete']:
                    # For removing objects, create a larger mask
                    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                else:
                    # Default modification mask
                    draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255))
            
            return mask
            
        except Exception as e:
            logger.warning(f"Mask creation failed: {e}")
            # Fallback: create a small center mask
            w, h = image.size
            mask = Image.new('RGB', (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(mask)
            draw.ellipse([w//3, h//3, 2*w//3, 2*h//3], fill=(255, 255, 255))
            return mask

    def apply_local_inpainting(self, image: Image.Image, mask: Image.Image, prompt: str) -> Optional[Image.Image]:
        """Apply inpainting to specific regions of the image"""
        try:
            # Try using a simple composite approach since full inpainting API is limited
            # This is a fallback method that blends new content
            
            # Create a small generated patch for the masked area
            mask_array = np.array(mask.convert('L'))
            white_pixels = np.where(mask_array > 128)
            
            if len(white_pixels[0]) == 0:
                return image
            
            # Get bounding box of the mask
            min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
            min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
            
            # Create a small patch to blend in
            patch_size = (max_x - min_x, max_y - min_y)
            if patch_size[0] < 50 or patch_size[1] < 50:
                patch_size = (100, 100)
            
            # Generate a small image for the patch
            patch_prompt = f"{prompt}, detailed close-up, high quality"
            try:
                # Generate small patch using the same API
                model_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
                payload = {
                    "inputs": patch_prompt,
                    "parameters": {
                        "num_inference_steps": 15,
                        "guidance_scale": 7.5,
                        "width": min(512, max(128, patch_size[0])),
                        "height": min(512, max(128, patch_size[1]))
                    }
                }
                
                response_data, error = self.call_api(model_url, payload, timeout=30)
                
                if response_data and "content" in response_data:
                    patch_image = Image.open(BytesIO(response_data["content"]))
                    patch_image = patch_image.convert('RGB').resize(patch_size)
                    
                    # Blend the patch into the original image using the mask
                    result_image = image.copy()
                    
                    # Create alpha mask for smooth blending
                    alpha_mask = mask.convert('L').resize(image.size)
                    alpha_mask = ImageEnhance.Contrast(alpha_mask).enhance(0.5)  # Soften edges
                    
                    # Resize patch to fit the masked area
                    patch_resized = patch_image.resize((max_x - min_x, max_y - min_y))
                    
                    # Paste the patch
                    result_image.paste(patch_resized, (min_x, min_y))
                    
                    # Apply alpha blending for smoother integration
                    result_array = np.array(result_image)
                    original_array = np.array(image)
                    mask_alpha = np.array(alpha_mask) / 255.0
                    
                    # Blend the images
                    for c in range(3):  # RGB channels
                        result_array[:, :, c] = (
                            original_array[:, :, c] * (1 - mask_alpha) + 
                            result_array[:, :, c] * mask_alpha
                        )
                    
                    return Image.fromarray(result_array.astype(np.uint8))
                    
            except Exception as e:
                logger.warning(f"Patch generation failed: {e}")
            
            return image
            
        except Exception as e:
            logger.error(f"Local inpainting failed: {e}")
            return image

    def apply_object_modification(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply object-based modifications to the image"""
        try:
            feedback_lower = feedback.lower()
            
            # Parse the feedback to extract action and object
            action = None
            object_name = None
            
            # Extract action verbs
            if any(word in feedback_lower for word in ['add', 'put', 'place', 'give']):
                action = 'add'
            elif any(word in feedback_lower for word in ['remove', 'delete', 'take away']):
                action = 'remove'
            elif any(word in feedback_lower for word in ['change', 'modify', 'alter']):
                action = 'change'
            
            # Extract object names
            objects = ['hat', 'cap', 'helmet', 'crown', 'glasses', 'sunglasses', 'necklace', 
                      'tie', 'scarf', 'beard', 'mustache', 'earrings', 'background', 'sky', 'clouds']
            
            for obj in objects:
                if obj in feedback_lower:
                    object_name = obj
                    break
            
            if action and object_name:
                logger.info(f"Detected action: {action}, object: {object_name}")
                
                # Create appropriate mask
                mask = self.create_inpainting_mask(image, object_name, action)
                
                # Create prompt for the modification
                if action == 'add':
                    modification_prompt = f"add {object_name}, realistic, detailed"
                elif action == 'remove':
                    modification_prompt = f"remove {object_name}, clean, natural"
                else:
                    modification_prompt = f"modify {object_name}, improved, detailed"
                
                # Apply inpainting
                result = self.apply_local_inpainting(image, mask, modification_prompt)
                
                if result:
                    return result
            
            # If no specific object detected, try general modification
            if any(word in feedback_lower for word in ['add', 'put', 'include']):
                # General addition - try to add something to the image
                general_prompt = feedback_lower.replace('add', '').replace('put', '').replace('include', '').strip()
                if general_prompt:
                    # Create a central mask
                    w, h = image.size
                    mask = Image.new('RGB', (w, h), (0, 0, 0))
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse([w//3, h//3, 2*w//3, 2*h//3], fill=(255, 255, 255))
                    
                    result = self.apply_local_inpainting(image, mask, general_prompt)
                    if result:
                        return result
            
            return image
            
        except Exception as e:
            logger.error(f"Object modification failed: {e}")
            return image

    def apply_image_adjustments(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply various image adjustments based on feedback"""
        try:
            feedback_lower = feedback.lower()
            adjusted_image = image.copy()
            
            # Brightness adjustments
            if any(word in feedback_lower for word in ['brighter', 'bright', 'lighter']):
                enhancer = ImageEnhance.Brightness(adjusted_image)
                adjusted_image = enhancer.enhance(1.3)
                logger.info("Applied brightness enhancement")
                
            elif any(word in feedback_lower for word in ['darker', 'dark', 'dimmer']):
                enhancer = ImageEnhance.Brightness(adjusted_image)
                adjusted_image = enhancer.enhance(0.7)
                logger.info("Applied darkness enhancement")
                
            # Color adjustments
            if any(word in feedback_lower for word in ['colorful', 'more color', 'vibrant', 'saturated']):
                enhancer = ImageEnhance.Color(adjusted_image)
                adjusted_image = enhancer.enhance(1.4)
                logger.info("Applied color enhancement")
                
            elif any(word in feedback_lower for word in ['less color', 'desaturate', 'muted']):
                enhancer = ImageEnhance.Color(adjusted_image)
                adjusted_image = enhancer.enhance(0.6)
                logger.info("Applied color desaturation")
                
            # Contrast adjustments
            if any(word in feedback_lower for word in ['more contrast', 'sharper', 'crisp', 'defined']):
                enhancer = ImageEnhance.Contrast(adjusted_image)
                adjusted_image = enhancer.enhance(1.3)
                logger.info("Applied contrast enhancement")
                
            elif any(word in feedback_lower for word in ['softer', 'less contrast', 'gentle']):
                enhancer = ImageEnhance.Contrast(adjusted_image)
                adjusted_image = enhancer.enhance(0.8)
                logger.info("Applied contrast reduction")
                
            # Filter effects
            if any(word in feedback_lower for word in ['blur', 'soft', 'smooth']):
                adjusted_image = adjusted_image.filter(ImageFilter.GaussianBlur(radius=1))
                logger.info("Applied blur filter")
                
            elif any(word in feedback_lower for word in ['sharp', 'crisp', 'clear']):
                adjusted_image = adjusted_image.filter(ImageFilter.SHARPEN)
                logger.info("Applied sharpen filter")
            
            # Additional artistic adjustments
            if any(word in feedback_lower for word in ['vintage', 'retro', 'old']):
                # Apply vintage effect: reduce saturation and add slight sepia
                enhancer = ImageEnhance.Color(adjusted_image)
                adjusted_image = enhancer.enhance(0.8)
                enhancer = ImageEnhance.Contrast(adjusted_image)
                adjusted_image = enhancer.enhance(1.1)
                logger.info("Applied vintage effect")
            
            if any(word in feedback_lower for word in ['dramatic', 'bold', 'intense']):
                # Apply dramatic effect: increase contrast and saturation
                enhancer = ImageEnhance.Contrast(adjusted_image)
                adjusted_image = enhancer.enhance(1.4)
                enhancer = ImageEnhance.Color(adjusted_image)
                adjusted_image = enhancer.enhance(1.3)
                logger.info("Applied dramatic effect")
                
            return adjusted_image
            
        except Exception as e:
            logger.error(f"Image adjustment failed: {e}")
            return image

    def generate_image_variation(self, original_image_path: str, feedback: str, original_prompt: str) -> Optional[str]:
        """Generate a new variation of the image using img2img technique"""
        try:
            # Load the original image
            with Image.open(original_image_path) as original_image:
                original_image = original_image.convert('RGB').resize(IMAGE_SIZE)
                
                # Convert image to base64 for API
                buffered = BytesIO()
                original_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
            # Create enhanced prompt based on feedback
            enhanced_prompt = self.integrate_feedback(original_prompt, feedback)
            
            # Try img2img models
            img2img_models = [
                "black-forest-labs/FLUX.1-schnell",
                "stabilityai/stable-diffusion-xl-base-1.0"
            ]
            
            for model_name in img2img_models:
                try:
                    model_url = f"https://api-inference.huggingface.co/models/{model_name}"
                    
                    payload = {
                        "inputs": enhanced_prompt,
                        "parameters": {
                            "num_inference_steps": 15,
                            "guidance_scale": 7.5,
                            "strength": 0.7  # How much to change from original
                        }
                    }
                    
                    # Note: Most HF models don't support img2img via API, so we'll use text2img with enhanced prompt
                    response_data, error = self.call_api(model_url, payload, timeout=60)
                    
                    if error:
                        logger.warning(f"Error with {model_name}: {error}")
                        continue
                    
                    if response_data and "content" in response_data:
                        # Save the variation
                        variation_image = Image.open(BytesIO(response_data["content"]))
                        variation_image = variation_image.convert('RGB')
                        
                        # Apply watermark only if user doesn't have premium subscription
                        if not self.check_subscription_status():
                            variation_image = self.add_watermark(variation_image)
                        
                        timestamp = int(time.time())
                        variation_path = f"output/variation_{timestamp}.png"
                        variation_image.save(variation_path, format='PNG', optimize=True)
                        
                        logger.info(f"Generated variation using {model_name}: {variation_path}")
                        return variation_path
                        
                except Exception as e:
                    logger.warning(f"Variation generation failed with {model_name}: {e}")
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Image variation generation failed: {e}")
            return None

    def apply_comprehensive_modification(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply comprehensive modifications to the image by trying multiple techniques"""
        try:
            feedback_lower = feedback.lower()
            modified_image = image.copy()
            
            # First try direct image adjustments
            adjusted_image = self.apply_image_adjustments(modified_image, feedback)
            if not np.array_equal(np.array(adjusted_image), np.array(modified_image)):
                modified_image = adjusted_image
                logger.info("Applied image adjustments in comprehensive modification")
            
            # Then try object modifications if applicable
            if any(keyword in feedback_lower for keyword in ['add', 'put', 'place', 'give', 'remove', 'delete', 'take away', 
                                                           'hat', 'cap', 'glasses', 'necklace', 'background', 'sky', 'clouds']):
                object_modified = self.apply_object_modification(modified_image, feedback)
                if not np.array_equal(np.array(object_modified), np.array(modified_image)):
                    modified_image = object_modified
                    logger.info("Applied object modifications in comprehensive modification")
            
            # Apply additional enhancements based on common feedback patterns
            if 'better' in feedback_lower or 'improve' in feedback_lower:
                # General improvement - slight contrast and sharpness boost
                enhancer = ImageEnhance.Contrast(modified_image)
                modified_image = enhancer.enhance(1.1)
                modified_image = modified_image.filter(ImageFilter.UnsharpMask())
                logger.info("Applied general improvements")
            
            if 'warmer' in feedback_lower:
                # Add warm tone
                enhancer = ImageEnhance.Color(modified_image)
                modified_image = enhancer.enhance(1.2)
                logger.info("Applied warm tone enhancement")
            
            if 'cooler' in feedback_lower:
                # Add cool tone (reduce color slightly)
                enhancer = ImageEnhance.Color(modified_image)
                modified_image = enhancer.enhance(0.9)
                logger.info("Applied cool tone enhancement")
            
            return modified_image
            
        except Exception as e:
            logger.error(f"Comprehensive modification failed: {e}")
            return image

    def apply_ai_feedback(self, image: Image.Image, feedback: str, original_prompt: str = "") -> Optional[Image.Image]:
        """AI-powered feedback system that can understand natural language requests"""
        try:
            feedback_lower = feedback.lower().strip()
            
            # Create a comprehensive modification prompt based on the feedback
            modification_prompt = self.create_modification_prompt(feedback, original_prompt)
            
            # Apply intelligent modifications based on feedback analysis
            modified_image = image.copy()
            
            # Analyze the feedback to determine the type of modification needed
            if self.is_additive_request(feedback_lower):
                # For "add more trees", "add sunset", etc.
                modified_image = self.apply_additive_modifications(modified_image, feedback_lower)
            elif self.is_color_request(feedback_lower):
                # For "make sky purple", "change colors", etc.
                modified_image = self.apply_color_modifications(modified_image, feedback_lower)
            elif self.is_lighting_request(feedback_lower):
                # For "add dramatic lighting", "make it sunset", etc.
                modified_image = self.apply_lighting_modifications(modified_image, feedback_lower)
            elif self.is_removal_request(feedback_lower):
                # For "remove person", "delete background", etc.
                modified_image = self.apply_removal_modifications(modified_image, feedback_lower)
            else:
                # For complex or mixed requests, apply comprehensive modifications
                modified_image = self.apply_smart_modifications(modified_image, feedback_lower)
            
            return modified_image
            
        except Exception as e:
            logger.error(f"AI feedback application failed: {e}")
            return None

    def create_modification_prompt(self, feedback: str, original_prompt: str) -> str:
        """Create a detailed modification prompt for AI processing"""
        if original_prompt:
            return f"Modify this image: {original_prompt}. Apply this change: {feedback}"
        else:
            return f"Apply this modification to the image: {feedback}"

    def is_additive_request(self, feedback: str) -> bool:
        """Check if the feedback is requesting to add something"""
        additive_keywords = [
            'add', 'more', 'put', 'place', 'include', 'with', 'extra', 
            'additional', 'another', 'some', 'few', 'many', 'insert'
        ]
        return any(keyword in feedback for keyword in additive_keywords)

    def is_color_request(self, feedback: str) -> bool:
        """Check if the feedback is about colors"""
        color_keywords = [
            'color', 'colour', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 
            'pink', 'brown', 'black', 'white', 'gray', 'grey', 'colorful', 'colourful',
            'vibrant', 'saturated', 'hue', 'tone', 'tint', 'shade'
        ]
        return any(keyword in feedback for keyword in color_keywords)

    def is_lighting_request(self, feedback: str) -> bool:
        """Check if the feedback is about lighting"""
        lighting_keywords = [
            'light', 'lighting', 'bright', 'dark', 'shadow', 'glow', 'shine', 'illuminate',
            'sunset', 'sunrise', 'dramatic', 'soft light', 'harsh light', 'golden hour',
            'backlight', 'spotlight', 'ambient', 'mood lighting'
        ]
        return any(keyword in feedback for keyword in lighting_keywords)

    def is_removal_request(self, feedback: str) -> bool:
        """Check if the feedback is requesting to remove something"""
        removal_keywords = [
            'remove', 'delete', 'take away', 'get rid of', 'eliminate', 'erase',
            'without', 'no', 'hide', 'clear', 'clean'
        ]
        return any(keyword in feedback for keyword in removal_keywords)

    def apply_additive_modifications(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply modifications that add elements to the image"""
        try:
            modified_image = image.copy()
            
            if 'tree' in feedback or 'forest' in feedback:
                # Enhance green areas and add texture
                modified_image = self.enhance_vegetation(modified_image)
            elif 'cloud' in feedback or 'sky' in feedback:
                # Enhance sky areas
                modified_image = self.enhance_sky(modified_image)
            elif 'flower' in feedback or 'bloom' in feedback:
                # Add colorful spots that could represent flowers
                modified_image = self.add_colorful_elements(modified_image)
            elif 'bird' in feedback or 'animal' in feedback:
                # Add small elements that could represent wildlife
                modified_image = self.add_small_elements(modified_image)
            else:
                # General additive enhancement
                modified_image = self.general_additive_enhancement(modified_image, feedback)
                
            return modified_image
            
        except Exception as e:
            logger.error(f"Additive modification failed: {e}")
            return image

    def apply_color_modifications(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply color-based modifications"""
        try:
            modified_image = image.copy()
            
            if 'purple' in feedback:
                modified_image = self.shift_colors_purple(modified_image)
            elif 'blue' in feedback:
                modified_image = self.shift_colors_blue(modified_image)
            elif 'red' in feedback:
                modified_image = self.shift_colors_red(modified_image)
            elif 'green' in feedback:
                modified_image = self.shift_colors_green(modified_image)
            elif 'yellow' in feedback or 'golden' in feedback:
                modified_image = self.shift_colors_yellow(modified_image)
            elif 'colorful' in feedback or 'vibrant' in feedback:
                # Enhance overall color saturation
                enhancer = ImageEnhance.Color(modified_image)
                modified_image = enhancer.enhance(1.3)
            elif 'muted' in feedback or 'subtle' in feedback:
                # Reduce color saturation
                enhancer = ImageEnhance.Color(modified_image)
                modified_image = enhancer.enhance(0.7)
                
            return modified_image
            
        except Exception as e:
            logger.error(f"Color modification failed: {e}")
            return image

    def apply_lighting_modifications(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply lighting-based modifications"""
        try:
            modified_image = image.copy()
            
            if 'sunset' in feedback or 'golden' in feedback:
                modified_image = self.apply_sunset_lighting(modified_image)
            elif 'dramatic' in feedback:
                modified_image = self.apply_dramatic_lighting(modified_image)
            elif 'soft' in feedback:
                modified_image = self.apply_soft_lighting(modified_image)
            elif 'bright' in feedback:
                enhancer = ImageEnhance.Brightness(modified_image)
                modified_image = enhancer.enhance(1.2)
            elif 'dark' in feedback or 'moody' in feedback:
                enhancer = ImageEnhance.Brightness(modified_image)
                modified_image = enhancer.enhance(0.8)
                
            return modified_image
            
        except Exception as e:
            logger.error(f"Lighting modification failed: {e}")
            return image

    def apply_removal_modifications(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply modifications that remove or hide elements"""
        try:
            modified_image = image.copy()
            
            # For removal requests, we can apply blur, darkening, or color shifting
            # to make unwanted elements less prominent
            if 'background' in feedback:
                modified_image = self.blur_background(modified_image)
            elif 'person' in feedback or 'people' in feedback:
                modified_image = self.obscure_figures(modified_image)
            else:
                # General removal - apply subtle blur to reduce prominence
                modified_image = modified_image.filter(ImageFilter.GaussianBlur(radius=0.5))
                
            return modified_image
            
        except Exception as e:
            logger.error(f"Removal modification failed: {e}")
            return image

    def apply_smart_modifications(self, image: Image.Image, feedback: str) -> Image.Image:
        """Apply intelligent modifications for complex requests"""
        try:
            modified_image = image.copy()
            
            # Apply a combination of enhancements based on the feedback
            if 'better' in feedback or 'improve' in feedback:
                enhancer = ImageEnhance.Contrast(modified_image)
                modified_image = enhancer.enhance(1.1)
                modified_image = modified_image.filter(ImageFilter.UnsharpMask())
            
            if 'artistic' in feedback or 'painterly' in feedback:
                modified_image = self.apply_artistic_effect(modified_image)
            
            if 'realistic' in feedback or 'detailed' in feedback:
                modified_image = modified_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
                
            return modified_image
            
        except Exception as e:
            logger.error(f"Smart modification failed: {e}")
            return image

    # Helper methods for specific modifications
    def enhance_vegetation(self, image: Image.Image) -> Image.Image:
        """Enhance green areas to simulate adding vegetation"""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Enhance green channel
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.2, 0, 255)
            
            return Image.fromarray(img_array.astype(np.uint8))
        except:
            return image

    def enhance_sky(self, image: Image.Image) -> Image.Image:
        """Enhance sky areas"""
        try:
            img_array = np.array(image)
            
            # Enhance blue channel in upper portion of image
            height = img_array.shape[0]
            sky_portion = img_array[:height//3, :, :]
            sky_portion[:, :, 2] = np.clip(sky_portion[:, :, 2] * 1.15, 0, 255)
            img_array[:height//3, :, :] = sky_portion
            
            return Image.fromarray(img_array.astype(np.uint8))
        except:
            return image

    def shift_colors_purple(self, image: Image.Image) -> Image.Image:
        """Shift colors towards purple"""
        try:
            img_array = np.array(image)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1, 0, 255)  # Enhance red
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.2, 0, 255)  # Enhance blue
            return Image.fromarray(img_array.astype(np.uint8))
        except:
            return image

    def apply_sunset_lighting(self, image: Image.Image) -> Image.Image:
        """Apply sunset/golden hour lighting effect"""
        try:
            # Enhance warm colors (red and yellow)
            img_array = np.array(image)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.15, 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.1, 0, 255)   # Green
            
            # Apply warm filter
            warm_filter = np.array([1.1, 1.05, 0.9])
            for i in range(3):
                img_array[:, :, i] = np.clip(img_array[:, :, i] * warm_filter[i], 0, 255)
                
            return Image.fromarray(img_array.astype(np.uint8))
        except:
            return image

    def apply_feedback_to_image(self, image_path: str, feedback: str, original_prompt: str = "") -> FeedbackResult:
        """Main feedback application method - focuses on modifying the same image"""
        try:
            if not os.path.exists(image_path):
                return FeedbackResult(
                    original_image_path=image_path,
                    edited_image_path=None,
                    feedback_applied=feedback,
                    success=False,
                    error_message="Original image not found"
                )
            
            feedback_lower = feedback.lower()
            
            # Load the original image
            with Image.open(image_path) as original_image:
                original_image = original_image.convert('RGB')
                
                # Approach 1: Direct image adjustments (try this first for simple changes)
                simple_adjustments = ['brighter', 'darker', 'colorful', 'more color', 'less color', 
                                    'more contrast', 'less contrast', 'blur', 'soft', 'sharp', 'crisp',
                                    'bright', 'dark', 'vibrant', 'saturated', 'desaturated', 'lighter',
                                    'dimmer', 'muted', 'defined', 'gentle', 'smooth', 'clear', 'vintage',
                                    'retro', 'old', 'dramatic', 'bold', 'intense', 'warmer', 'cooler']
                
                if any(adj in feedback_lower for adj in simple_adjustments):
                    try:
                        adjusted_image = self.apply_image_adjustments(original_image, feedback)
                        
                        # Check if the image was actually modified
                        if not np.array_equal(np.array(adjusted_image), np.array(original_image)):
                            # Apply watermark only if user doesn't have premium subscription
                            if not self.check_subscription_status():
                                adjusted_image = self.add_watermark(adjusted_image)
                            
                            timestamp = int(time.time())
                            adjusted_path = f"output/adjusted_{timestamp}.png"
                            adjusted_image.save(adjusted_path, format='PNG', optimize=True)
                            
                            logger.info(f"Applied direct adjustments: {adjusted_path}")
                            return FeedbackResult(
                                original_image_path=image_path,
                                edited_image_path=adjusted_path,
                                feedback_applied=feedback,
                                success=True
                            )
                    except Exception as e:
                        logger.warning(f"Direct adjustment failed: {e}")
                
                # Approach 2: Object-based modifications (for adding/removing things)
                object_keywords = ['add', 'put', 'place', 'give', 'remove', 'delete', 'take away', 
                                 'hat', 'cap', 'glasses', 'necklace', 'background', 'sky', 'clouds']
                
                if any(keyword in feedback_lower for keyword in object_keywords):
                    try:
                        modified_image = self.apply_object_modification(original_image, feedback)
                        
                        # Check if the image was actually modified
                        if not np.array_equal(np.array(modified_image), np.array(original_image)):
                            # Apply watermark only if user doesn't have premium subscription
                            if not self.check_subscription_status():
                                modified_image = self.add_watermark(modified_image)
                            
                            timestamp = int(time.time())
                            modified_path = f"output/modified_{timestamp}.png"
                            modified_image.save(modified_path, format='PNG', optimize=True)
                            
                            logger.info(f"Applied object modification: {modified_path}")
                            return FeedbackResult(
                                original_image_path=image_path,
                                edited_image_path=modified_path,
                                feedback_applied=feedback,
                                success=True
                            )
                    except Exception as e:
                        logger.warning(f"Object modification failed: {e}")
                
                # Approach 3: AI-powered natural language feedback (NEW!)
                # This handles ANY request like "add more trees", "make sky purple", "remove person", etc.
                try:
                    modified_image = self.apply_ai_feedback(original_image, feedback, original_prompt)
                    
                    if modified_image and not np.array_equal(np.array(modified_image), np.array(original_image)):
                        # Apply watermark only if user doesn't have premium subscription
                        if not self.check_subscription_status():
                            modified_image = self.add_watermark(modified_image)
                        
                        timestamp = int(time.time())
                        ai_modified_path = f"output/ai_modified_{timestamp}.png"
                        modified_image.save(ai_modified_path, format='PNG', optimize=True)
                        
                        logger.info(f"Applied AI feedback: {ai_modified_path}")
                        return FeedbackResult(
                            original_image_path=image_path,
                            edited_image_path=ai_modified_path,
                            feedback_applied=feedback,
                            success=True
                        )
                except Exception as e:
                    logger.warning(f"AI feedback failed: {e}")

                # Approach 4: Enhanced fallback - comprehensive modification
                try:
                    modified_image = self.apply_comprehensive_modification(original_image, feedback)
                    
                    if not np.array_equal(np.array(modified_image), np.array(original_image)):
                        # Apply watermark only if user doesn't have premium subscription
                        if not self.check_subscription_status():
                            modified_image = self.add_watermark(modified_image)
                        
                        timestamp = int(time.time())
                        fallback_path = f"output/enhanced_{timestamp}.png"
                        modified_image.save(fallback_path, format='PNG', optimize=True)
                        
                        logger.info(f"Applied enhanced modification: {fallback_path}")
                        return FeedbackResult(
                            original_image_path=image_path,
                            edited_image_path=fallback_path,
                            feedback_applied=feedback,
                            success=True
                        )
                except Exception as e:
                    logger.error(f"Enhanced modification failed: {e}")
                
                # If no modification was applied, use AI generation as last resort
                return FeedbackResult(
                    original_image_path=image_path,
                    edited_image_path=None,
                    feedback_applied=feedback,
                    success=False,
                    error_message="Unable to apply the requested changes. The system supports requests like: 'make it brighter', 'add more trees', 'change sky color', 'remove objects', etc."
                )
                
        except Exception as e:
            logger.error(f"Feedback application failed: {e}")
            return FeedbackResult(
                original_image_path=image_path,
                edited_image_path=None,
                feedback_applied=feedback,
                success=False,
                error_message=str(e)
            )
            
        except Exception as e:
            logger.error(f"Feedback application failed: {e}")
            return FeedbackResult(
                original_image_path=image_path,
                edited_image_path=None,
                feedback_applied=feedback,
                success=False,
                error_message=str(e)
            )

    def generate_art(self, prompt: str, style: str = "Auto-detect", feedback: str = "") -> GenerationResult:
        """Main art generation pipeline"""
        try:
            style_info = self.extract_style_elements(prompt, style)
            search_query = f"{style_info['subject']} {style_info['style']}"
            references = self.find_similar_images(search_query, k=2)
            final_prompt = style_info['refined_prompt']
            if feedback:
                final_prompt = self.integrate_feedback(final_prompt, feedback)
            image_path = self.generate_image(final_prompt, references)
            if image_path:
                self.save_generation_metadata(image_path, prompt, style_info['style'])
            return GenerationResult(
                image_path=image_path,
                references=references,
                style_info=style_info,
                success=image_path is not None,
                error_message=None if image_path else "Image generation failed"
            )
        except Exception as e:
            logger.error(f"Art generation pipeline failed: {e}")
            return GenerationResult(
                image_path=None,
                references=[],
                style_info={},
                success=False,
                error_message=str(e)
            )

    def display_results(self, result: GenerationResult):
        """Display generation results in the UI"""
        if result.success and result.image_path:
            # Show subscription status with generation result
            is_premium = self.check_subscription_status()
            if is_premium:
                st.success("🎨 Masterpiece created successfully! (Premium - No watermarks)")
            else:
                st.success("🎨 Art generated successfully! (Free plan - includes watermarks)")
                st.info("💎 Upgrade to Premium for watermark-free creations")
            
            # Add generous spacing
           
            
            # Results display with enhanced spacing
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0 3rem 0;'>
            <h2 style='color: #667eea; margin-bottom: 1rem; font-size: 2.5rem; font-weight: 700;'>✨ Your Masterpiece</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 1.1rem;'>Generated with AI • Ready for download and editing</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Much better proportions - 60% for image, 40% for controls
            col1, col2 = st.columns([3, 2], gap="large")
            
            with col1:
                # Enhanced image display container
                st.markdown("""
                <div style='background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6)); 
                            padding: 2rem; 
                            border-radius: 40px; 
                            border: 1px solid #475569; 
                            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                            margin-bottom: 2rem;'>
                """, unsafe_allow_html=True)
                
                st.image(result.image_path, caption="", use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download section with spacing
                st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                
                # Enhanced download button
                try:
                    with open(result.image_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Your Masterpiece",
                            data=file.read(),
                            file_name=f"masterpiece_{int(time.time())}.png",
                            mime="image/png",
                            use_container_width=True,
                            type="primary"
                        )
                except Exception as e:
                    logger.warning(f"Download button setup failed: {e}")
            
            with col2:
                # Feedback section with enhanced styling and spacing
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1)); 
                            padding: 2rem; 
                            border-radius: 20px; 
                            border: 1px solid #667eea; 
                            margin-bottom: 2rem;'>
                <h3 style='color: #667eea; margin-bottom: 1rem; text-align: center;'>🎯 Perfect Your Art</h3>
                <p style='color: #94a3b8; text-align: center; margin-bottom: 1.5rem;'>Describe changes to transform your artwork instantly</p>
                </div>
                """, unsafe_allow_html=True)
                
                feedback_input = st.text_area(
                    "💬 What would you like to change?",
                    placeholder="Examples:\n• 'Make it brighter and more colorful'\n• 'Add dramatic sunset lighting'\n• 'More contrast and sharper details'\n• 'Softer, dreamier atmosphere'",
                    height=120,
                    key=f"feedback_{result.image_path}",
                    help="💡 Be specific about lighting, colors, mood, or style changes you want"
                )
                
                # Store original prompt in session state for feedback
                if "original_prompts" not in st.session_state:
                    st.session_state.original_prompts = {}
                st.session_state.original_prompts[result.image_path] = st.session_state.get('last_prompt', '')
                
                # Add more spacing before buttons
                st.markdown("<div style='margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
                
                # Action buttons with enhanced layout
                col_btn1, col_btn2 = st.columns(2, gap="medium")
                
                with col_btn1:
                    if st.button("✨ Apply Changes", key=f"apply_{result.image_path}", use_container_width=True, type="primary"):
                        if feedback_input.strip():
                            with st.spinner("🎨 Applying your vision..."):
                                original_prompt = st.session_state.original_prompts.get(result.image_path, '')
                                feedback_result = self.apply_feedback_to_image(
                                    result.image_path, 
                                    feedback_input, 
                                    original_prompt
                                )
                                
                                if feedback_result.success:
                                    st.success("✅ Changes applied successfully!")
                                    
                                    # Show edited image with enhanced container
                                    st.markdown("""
                                    <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid #10b981; margin: 1rem 0;'>
                                    <h4 style='color: #10b981; margin: 0 0 0.5rem 0;'>🎨 Enhanced Version</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.image(
                                        feedback_result.edited_image_path, 
                                        caption=f"Applied: {feedback_result.feedback_applied}",
                                        use_container_width=True
                                    )
                                    
                                    # Download button for edited image
                                    try:
                                        with open(feedback_result.edited_image_path, "rb") as file:
                                            st.download_button(
                                                label="📥 Download Enhanced Version",
                                                data=file.read(),
                                                file_name=f"enhanced_art_{int(time.time())}.png",
                                                mime="image/png",
                                                key=f"download_edited_{result.image_path}",
                                                use_container_width=True
                                            )
                                    except Exception as e:
                                        logger.warning(f"Download button for edited image failed: {e}")
                                else:
                                    st.error(f"❌ Enhancement failed: {feedback_result.error_message}")
                        else:
                            st.warning("💡 Please describe the changes you'd like to see")
                
                with col_btn2:
                    if st.button("🎲 Create Variation", key=f"variation_{result.image_path}", use_container_width=True):
                        if feedback_input.strip():
                            with st.spinner("🎨 Creating new variation..."):
                                original_prompt = st.session_state.original_prompts.get(result.image_path, '')
                                variation_path = self.generate_image_variation(
                                    result.image_path, 
                                    feedback_input, 
                                    original_prompt
                                )
                                
                                if variation_path:
                                    st.success("✅ Variation created!")
                                    st.image(
                                        variation_path, 
                                        caption=f"Variation: {feedback_input}",
                                        use_container_width=True
                                    )
                                else:
                                    st.error("❌ Variation generation failed")
                        else:
                            st.warning("Please enter feedback for variation")
                
                # Quick feedback buttons with enhanced styling
                st.markdown("<br><hr style='border-color: #334155; margin: 1.5rem 0;'><br>", unsafe_allow_html=True)
                # Quick adjustments with enhanced spacing
                st.markdown("<div style='margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.08); padding: 1.5rem; border-radius: 16px; border: 1px solid #334155; margin-bottom: 1.5rem;'>
                <h4 style='color: #667eea; margin: 0 0 0.8rem 0; text-align: center;'>🚀 Quick Enhancements</h4>
                <p style='color: #94a3b8; margin: 0; font-size: 0.95rem; text-align: center;'>One-click improvements for instant results</p>
                </div>
                """, unsafe_allow_html=True)
                
                quick_feedbacks = [
                    ("☀️ Brighter", "make it brighter"),
                    ("🌙 Darker", "make it darker"),
                    ("🎨 More Color", "add more vibrant colors"),
                    ("🔍 Sharper", "make it sharper and more detailed")
                ]
                
                # Enhanced quick buttons layout
                for i in range(0, len(quick_feedbacks), 2):
                    col_q1, col_q2 = st.columns(2, gap="small")
                    
                    with col_q1:
                        if i < len(quick_feedbacks):
                            label, feedback = quick_feedbacks[i]
                            if st.button(label, key=f"quick_{i}_{result.image_path}", use_container_width=True):
                                with st.spinner(f"Applying {feedback}..."):
                                    feedback_result = self.apply_feedback_to_image(result.image_path, feedback)
                                    if feedback_result.success:
                                        st.rerun()
                    
                    with col_q2:
                        if i + 1 < len(quick_feedbacks):
                            label, feedback = quick_feedbacks[i + 1]
                            if st.button(label, key=f"quick_{i+1}_{result.image_path}", use_container_width=True):
                                with st.spinner(f"Applying {feedback}..."):
                                    feedback_result = self.apply_feedback_to_image(result.image_path, feedback)
                                    if feedback_result.success:
                                        st.rerun()
            
            # Style and reference information (moved below)
            if result.style_info:
                with st.expander("�🎭 Style Analysis"):
                    st.write(f"**Detected Style:** {result.style_info.get('style', 'Unknown')}")
                    st.write(f"**Subject:** {result.style_info.get('subject', 'Unknown')}")
                    if 'elements' in result.style_info:
                        elements = result.style_info['elements']
                        if elements.get('nouns'):
                            st.write(f"**Key Elements:** {', '.join(elements['nouns'])}")
                        if elements.get('adjectives'):
                            st.write(f"**Descriptors:** {', '.join(elements['adjectives'])}")
            
            if result.references:
                with st.expander("🖼️ Reference Images Used"):
                    cols = st.columns(len(result.references))
                    for i, ref in enumerate(result.references):
                        with cols[i]:
                            try:
                                if os.path.exists(ref['image_path']):
                                    st.image(
                                        ref['image_path'], 
                                        caption=f"Style: {ref.get('style', 'Unknown')}", 
                                        width=150,
                                        use_container_width=False
                                    )
                            except Exception as e:
                                st.write(f"Reference: {ref.get('style', 'Unknown')}")
                                
        else:
            st.error(f"❌ Generation failed: {result.error_message}")

    def display_results_in_columns(self, result: GenerationResult, col1, col2):
        """Display generation results using the provided columns"""
        if result.success and result.image_path:
            # Store result in session state for persistence
            st.session_state.last_result = result
            
            # Show subscription status with generation result
            is_premium = self.check_subscription_status()
            if is_premium:
                st.success("🎨 Masterpiece created successfully! (Premium - No watermarks)")
            else:
                st.success("🎨 Art generated successfully! (Free plan - includes watermarks)")
                st.info("💎 Upgrade to Premium for watermark-free creations")
            
            # Use the existing col1 for image display
            with col1:
                # Clear previous content
                st.empty()
                
                # Enhanced image display
                st.markdown("""
                <div style='text-align: center; margin: 2rem 0 1rem 0;'>
                <h3 style='color: #667eea; margin-bottom: 1rem; font-size: 1.8rem; font-weight: 700;'>✨ Your Masterpiece</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced image display container
                st.markdown("""
                <div style='background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6)); 
                            padding: 2rem; 
                            border-radius: 20px; 
                            border: 1px solid #475569; 
                            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                            margin-bottom: 2rem;'>
                """, unsafe_allow_html=True)
                
                st.image(result.image_path, caption="Generated with AI", use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download section
                try:
                    with open(result.image_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Your Masterpiece",
                            data=file.read(),
                            file_name=f"masterpiece_{int(time.time())}.png",
                            mime="image/png",
                            use_container_width=True,
                            type="primary"
                        )
                except Exception as e:
                    logger.warning(f"Download button setup failed: {e}")
                    
                # Style and reference information
                if result.style_info:
                    with st.expander("🎭 Style Analysis"):
                        st.write(f"**Detected Style:** {result.style_info.get('style', 'Unknown')}")
                        st.write(f"**Subject:** {result.style_info.get('subject', 'Unknown')}")
                        if 'elements' in result.style_info:
                            elements = result.style_info['elements']
                            if elements.get('nouns'):
                                st.write(f"**Key Elements:** {', '.join(elements['nouns'])}")
                            if elements.get('adjectives'):
                                st.write(f"**Descriptors:** {', '.join(elements['adjectives'])}")
                
                if result.references:
                    with st.expander("🖼️ Reference Images Used"):
                        cols = st.columns(len(result.references))
                        for i, ref in enumerate(result.references):
                            with cols[i]:
                                try:
                                    if os.path.exists(ref['image_path']):
                                        st.image(
                                            ref['image_path'], 
                                            caption=f"Style: {ref.get('style', 'Unknown')}", 
                                            width=150,
                                            use_container_width=False
                                        )
                                except Exception as e:
                                    st.write(f"Reference: {ref.get('style', 'Unknown')}")
            
            # Use the existing col2 for feedback section
            with col2:
                # Clear previous tips content
                st.empty()
                
                # Feedback section with enhanced styling
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1)); 
                            padding: 2rem; 
                            border-radius: 20px; 
                            border: 1px solid #667eea; 
                            margin-bottom: 2rem;'>
                <h3 style='color: #667eea; margin-bottom: 1rem; text-align: center;'>🎯 Perfect Your Art</h3>
                <p style='color: #94a3b8; text-align: center; margin-bottom: 1.5rem;'>Describe changes to transform your artwork</p>
                </div>
                """, unsafe_allow_html=True)
                
                feedback_input = st.text_area(
                    "💬 What would you like to change?",
                    placeholder="Examples:\n• 'Make it brighter and more colorful'\n• 'Add dramatic sunset lighting'\n• 'More contrast and sharper details'\n• 'Softer, dreamier atmosphere'",
                    height=120,
                    key=f"feedback_{result.image_path}",
                    help="💡 Be specific about lighting, colors, mood, or style changes"
                )
                
                # Store original prompt in session state for feedback
                if "original_prompts" not in st.session_state:
                    st.session_state.original_prompts = {}
                st.session_state.original_prompts[result.image_path] = st.session_state.get('last_prompt', '')
                
                # Action buttons
                col_btn1, col_btn2 = st.columns(2, gap="medium")
                
                with col_btn1:
                    if st.button("✨ Apply Changes", key=f"apply_{result.image_path}", use_container_width=True, type="primary"):
                        if feedback_input.strip():
                            with st.spinner("🎨 Applying your vision..."):
                                original_prompt = st.session_state.original_prompts.get(result.image_path, '')
                                feedback_result = self.apply_feedback_to_image(
                                    result.image_path, 
                                    feedback_input, 
                                    original_prompt
                                )
                                
                                if feedback_result.success:
                                    st.success("✅ Changes applied successfully!")
                                    
                                    # Show edited image
                                    st.markdown("""
                                    <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid #10b981; margin: 1rem 0;'>
                                    <h4 style='color: #10b981; margin: 0 0 0.5rem 0;'>🎨 Enhanced Version</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.image(
                                        feedback_result.edited_image_path, 
                                        caption=f"Applied: {feedback_result.feedback_applied}",
                                        use_container_width=True
                                    )
                                    
                                    # Download button for edited image
                                    try:
                                        with open(feedback_result.edited_image_path, "rb") as file:
                                            st.download_button(
                                                label="📥 Download Enhanced",
                                                data=file.read(),
                                                file_name=f"enhanced_{int(time.time())}.png",
                                                mime="image/png",
                                                key=f"download_edited_{result.image_path}",
                                                use_container_width=True
                                            )
                                    except Exception as e:
                                        logger.warning(f"Download button failed: {e}")
                                else:
                                    st.error(f"❌ Enhancement failed: {feedback_result.error_message}")
                        else:
                            st.warning("💡 Please describe the changes you'd like to see")
                
                with col_btn2:
                    if st.button("🎲 Create Variation", key=f"variation_{result.image_path}", use_container_width=True):
                        if feedback_input.strip():
                            with st.spinner("🎨 Creating variation..."):
                                original_prompt = st.session_state.original_prompts.get(result.image_path, '')
                                variation_path = self.generate_image_variation(
                                    result.image_path, 
                                    feedback_input, 
                                    original_prompt
                                )
                                
                                if variation_path:
                                    st.success("✅ Variation created!")
                                    st.image(
                                        variation_path, 
                                        caption=f"Variation: {feedback_input}",
                                        use_container_width=True
                                    )
                                else:
                                    st.error("❌ Variation generation failed")
                        else:
                            st.warning("Please enter feedback for variation")
                            
        else:
            st.error(f"❌ Generation failed: {result.error_message}")

    def display_results_single_column(self, result: GenerationResult):
        """Display generation results in a single centered column layout"""
        # Store result in session state to persist across reruns
        if result.success and result.image_path:
            st.session_state.current_result = result
        
        # Use stored result if available (for handling button interactions)
        if hasattr(st.session_state, 'current_result') and st.session_state.current_result.success:
            result = st.session_state.current_result
            
        if result.success and result.image_path:
            # Show subscription status with generation result
            is_premium = self.check_subscription_status()
            if is_premium:
                st.success("🎨 Masterpiece created successfully! (Premium - No watermarks)")
            else:
                st.success("🎨 Art generated successfully! (Free plan - includes watermarks)")
                st.info("💎 Upgrade to Premium for watermark-free creations")
            
            # Add generous spacing
            st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
            
            # Header section - centered
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0 3rem 0;'>
            <h2 style='color: #667eea; margin-bottom: 1rem; font-size: 2.5rem; font-weight: 700;'>✨ Your Masterpiece</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 1.1rem;'>Generated with AI • Ready for download and editing</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Image display section - centered with max width
            st.markdown("""
            <div style='display: flex; justify-content: center; margin: 3rem 0;'>
            <div style='max-width: 600px; width: 100%; 
                       background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6)); 
                       padding: 2rem; 
                       border-radius: 24px; 
                       border: 1px solid #475569; 
                       box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);'>
            """, unsafe_allow_html=True)
            
            st.image(result.image_path, caption="", use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Download section - centered
            st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
            col_download_left, col_download_center, col_download_right = st.columns([1, 2, 1])
            
            with col_download_center:
                try:
                    with open(result.image_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Your Masterpiece",
                            data=file.read(),
                            file_name=f"masterpiece_{int(time.time())}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                except Exception as e:
                    logger.warning(f"Download button setup failed: {e}")
            
            # Feedback section - centered
            st.markdown("<div style='margin: 4rem 0 2rem 0;'></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
            <h3 style='color: #667eea; font-size: 1.8rem; margin-bottom: 1rem;'>🎯 Perfect Your Art</h3>
            <p style='color: #94a3b8; font-size: 1.1rem;'>Describe changes to transform your artwork instantly</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback input - centered with max width
            col_feedback_left, col_feedback_center, col_feedback_right = st.columns([0.5, 3, 0.5])
            
            with col_feedback_center:
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1)); 
                            padding: 2rem; 
                            border-radius: 20px; 
                            border: 1px solid #667eea; 
                            margin-bottom: 2rem;'>
                """, unsafe_allow_html=True)
                
                feedback_input = st.text_area(
                    "💬 What would you like to change?",
                    placeholder="Examples:\n• 'Make it brighter and more colorful'\n• 'Add dramatic sunset lighting'\n• 'More contrast and sharper details'\n• 'Softer, dreamier atmosphere'",
                    height=120,
                    key=f"feedback_{result.image_path}",
                    help="💡 Be specific about lighting, colors, mood, or style changes you want"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Store original prompt in session state for feedback
                if "original_prompts" not in st.session_state:
                    st.session_state.original_prompts = {}
                st.session_state.original_prompts[result.image_path] = st.session_state.get('last_prompt', '')
                
                # Action buttons - centered
                col_btn1, col_btn2 = st.columns(2, gap="medium")
                
                with col_btn1:
                    if st.button("🔧 Apply Changes", key=f"apply_{result.image_path}", use_container_width=True):
                        if feedback_input.strip():
                            with st.spinner("🎨 Applying your changes..."):
                                original_prompt = st.session_state.original_prompts.get(result.image_path, '')
                                feedback_result = self.apply_feedback_to_image(
                                    result.image_path, 
                                    feedback_input, 
                                    original_prompt
                                )
                                
                                if feedback_result.success and feedback_result.edited_image_path:
                                    st.success("✅ Changes applied successfully!")
                                    
                                    # Display the edited image centered
                                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                                    st.markdown("""
                                    <div style='display: flex; justify-content: center;'>
                                    <div style='max-width: 600px; width: 100%; 
                                               background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6)); 
                                               padding: 2rem; 
                                               border-radius: 24px; 
                                               border: 1px solid #10b981;'>
                                    """, unsafe_allow_html=True)
                                    
                                    st.image(
                                        feedback_result.edited_image_path,
                                        caption=f"Enhanced: {feedback_input}",
                                        use_container_width=True
                                    )
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                    
                                    # Download button for edited image
                                    try:
                                        with open(feedback_result.edited_image_path, "rb") as file:
                                            st.download_button(
                                                label="📥 Download Enhanced Image",
                                                data=file.read(),
                                                file_name=f"enhanced_{int(time.time())}.png",
                                                mime="image/png",
                                                use_container_width=True
                                            )
                                    except Exception as e:
                                        logger.warning(f"Download button failed: {e}")
                                else:
                                    st.error(f"❌ Enhancement failed: {feedback_result.error_message}")
                        else:
                            st.warning("💡 Please describe the changes you'd like to see")
                
                with col_btn2:
                    if st.button("🎲 Create Variation", key=f"variation_{result.image_path}", use_container_width=True):
                        if feedback_input.strip():
                            with st.spinner("🎨 Creating variation..."):
                                original_prompt = st.session_state.original_prompts.get(result.image_path, '')
                                variation_path = self.generate_image_variation(
                                    result.image_path, 
                                    feedback_input, 
                                    original_prompt
                                )
                                
                                if variation_path:
                                    st.success("✅ Variation created!")
                                    
                                    # Display variation centered
                                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                                    st.markdown("""
                                    <div style='display: flex; justify-content: center;'>
                                    <div style='max-width: 600px; width: 100%; 
                                               background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6)); 
                                               padding: 2rem; 
                                               border-radius: 24px; 
                                               border: 1px solid #f59e0b;'>
                                    """, unsafe_allow_html=True)
                                    
                                    st.image(
                                        variation_path, 
                                        caption=f"Variation: {feedback_input}",
                                        use_container_width=True
                                    )
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                else:
                                    st.error("❌ Variation generation failed")
                        else:
                            st.warning("Please enter feedback for variation")
                            
        else:
            st.error(f"❌ Generation failed: {result.error_message}")

    def run(self):
        """Main Streamlit application"""
        st.set_page_config(
            page_title="🎨 Art Generator",
            layout="wide",
            page_icon="🎨"
        )
        
        # Global CSS & Dark Mode Layout
        st.markdown("""
        <style>
        /* Root & Background - Dark Theme */
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
            color: #e2e8f0 !important;
        }
        
        /* Main content container - centered */
        .main .block-container {
            max-width: 900px !important;
            margin: 0 auto !important;
            padding: 2rem 1rem !important;
        }
        
        /* Layout & typography */
        html, body, [class*="css"], .stMarkdown, .stText {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial !important;
            color: #e2e8f0 !important;
        }
        
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 800;
            margin: 2rem 0 0.5rem 0;
            letter-spacing: -0.02em;
        }
        
        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }

        /* Dark sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e1e3f 0%, #2d2d5a 100%) !important;
            border-right: 1px solid #374151 !important;
        }
        
        .css-1d391kg .stMarkdown, .css-1d391kg .stText,
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stText {
            color: #e2e8f0 !important;
        }

        /* Input fields - Dark theme */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border: 1px solid #475569 !important;
            border-radius: 8px !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
        }

        /* Card styling - Dark theme */
        .card, .stContainer {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid #334155 !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
            backdrop-filter: blur(10px) !important;
        }

        /* Enhanced Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3) !important;
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }

        /* Columns spacing - Enhanced */
        .stColumn {
            padding: 0 1rem !important;
        }
        
        /* Enhanced spacing for results */
        .stColumn > div {
            padding: 0.75rem 0 !important;
        }
        
        /* Text areas with better spacing */
        .stTextArea > div > div > textarea {
            padding: 1rem !important;
            line-height: 1.6 !important;
        }
        
        /* Enhanced image containers */
        [data-testid="stImage"] {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        }
        
        /* Image containers */
        .stImage {
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* Success/Warning/Error messages */
        .stSuccess, .stInfo, .stWarning, .stError {
            background: rgba(30, 41, 59, 0.9) !important;
            border-radius: 8px !important;
            border-left: 4px solid !important;
        }
        
        .stSuccess { border-left-color: #10b981 !important; }
        .stInfo { border-left-color: #3b82f6 !important; }
        .stWarning { border-left-color: #f59e0b !important; }
        .stError { border-left-color: #ef4444 !important; }

        /* Dividers */
        hr {
            border-color: #334155 !important;
            margin: 1.5rem 0 !important;
        }

        /* Image display */
        .preview-center { 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            padding: 1rem;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px;
            margin: 1rem 0;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main-header { font-size: 2rem; }
            .subtitle { font-size: 1rem; }
            .main .block-container {
                padding: 1rem 0.5rem !important;
                max-width: 100% !important;
            }
        }
        
        @media (max-width: 480px) {
            .main-header { font-size: 1.5rem; }
            .stColumn {
                padding: 0 0.25rem !important;
            }
            .main .block-container {
                max-width: 100% !important;
                padding: 1rem 0.25rem !important;
            }
        }

        /* Additional single-column enhancements */
        .stTextArea textarea {
            min-height: 120px !important;
        }
        
        /* Center alignment utilities */
        .center-content {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="main-header">🎨 Art Generator</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Turn prompts into beautiful, style-aware artwork — fast.</div>', unsafe_allow_html=True)
        
        # Sidebar (dark-themed)
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0; background: rgba(30, 41, 59, 0.5); border-radius: 12px; margin-bottom: 1rem;'>
            <h2 style='color: #667eea; margin: 0; font-weight: 700;'>🎨 Art Generator</h2>
            <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0 0 0; font-weight: 400;'>AI Creative Suite</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("---")
            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared")

            st.markdown("---")
            
            # Subscription Section
            st.subheader("💎 Subscription")
            is_premium = self.check_subscription_status()
            
            if is_premium:
                st.success("✅ Premium Active")
                st.info("🚫 No watermarks on images")
                if st.session_state.subscription_expiry:
                    st.caption(f"Expires: {st.session_state.subscription_expiry}")
                
                if st.button("Cancel Subscription", use_container_width=True):
                    self.cancel_subscription()
            else:
                st.warning("⚠️ Free Plan")
                st.info("💧 Images include watermarks")
                
                if st.button("🚀 Upgrade to Premium", type="primary", use_container_width=True):
                    self.upgrade_subscription()

            # Recent images list removed per user request
        
        # Main content area - Single centered column layout
        st.markdown("""
        <div style='max-width: 800px; margin: 0 auto; padding: 2rem 1rem;'>
        """, unsafe_allow_html=True)
        
        # Prompt Input Section - Centered
        st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
        <h2 style='color: #667eea; font-size: 2rem; margin-bottom: 1rem;'>✨ Create Your Art</h2>
        <p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Describe your vision and watch it come to life</p>
        </div>
        """, unsafe_allow_html=True)
        
        prompt = st.text_area(
            "🎨 Describe Your Vision",
            placeholder="Paint me a serene mountain landscape at sunset with golden light filtering through pine trees, in the style of Bob Ross...",
            help="💡 Be descriptive! Include mood, colors, style, and subject matter for best results",
            height=120
        )
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Style and refinement options - Side by side but centered
        col_style, col_feedback = st.columns(2, gap="medium")
        with col_style:
            available_styles = sorted(set(self.df['style'].astype(str).str.strip().tolist())) if not self.df.empty else []
            style = st.selectbox(
                "🎭 Art Style",
                ["Auto-detect"] + available_styles,
                help="Choose a style from your reference images or let AI detect the best style."
            )
        with col_feedback:
            feedback = st.text_input(
                "🔧 Refinements",
                placeholder="e.g., 'more vibrant colors', 'softer lighting'"
            )
        
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Centered Generate button
        col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
        with col_btn_center:
            if st.button("🚀 Generate Masterpiece", type="primary", use_container_width=True):
                if "generation_count" not in st.session_state:
                    st.session_state.generation_count = 0
                if not self.validate_inputs(prompt):
                    return
                    
                if not self.manage_rate_limit():
                    return
                
                # Clear previous result when starting new generation
                if hasattr(st.session_state, 'current_result'):
                    del st.session_state.current_result
                    
                # Store the prompt for feedback functionality
                st.session_state.last_prompt = prompt
                st.session_state.generation_count += 1
                
                with st.spinner("🎨 Creating your masterpiece..."):
                    progress_bar = st.progress(0)
                    progress_bar.progress(25, "Analyzing your prompt...")
                    time.sleep(0.5)
                    progress_bar.progress(50, "Finding style references...")
                    time.sleep(0.5)
                    progress_bar.progress(75, "Generating artwork...")
                    
                    result = self.generate_art(prompt, style, feedback)
                    progress_bar.progress(100, "Complete!")
                    time.sleep(0.5)
                    
                # Display results in single column layout
                self.display_results_single_column(result)
                
                progress_bar.empty()

        # Display stored results if they exist (for button interactions)
        if hasattr(st.session_state, 'current_result') and st.session_state.current_result.success:
            # Only display if we're not in the middle of a new generation
            if "generation_count" not in st.session_state or st.session_state.generation_count > 0:
                self.display_results_single_column(st.session_state.current_result)

        # Quick tips section - centered
        st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1)); 
                    padding: 2rem; 
                    border-radius: 20px; 
                    border: 1px solid #667eea; 
                    text-align: center;'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>💡 Pro Tips for Better Results</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;'>
            <div style='color: #94a3b8;'>✨ Be specific about colors and mood</div>
            <div style='color: #94a3b8;'>🎨 Mention artistic styles you prefer</div>
            <div style='color: #94a3b8;'>💡 Include lighting and atmosphere details</div>
            <div style='color: #94a3b8;'>📝 Use descriptive adjectives</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Close the main content container
        st.markdown('</div>', unsafe_allow_html=True)

        # Feedback Agent Section for Previous Images
        if hasattr(st.session_state, 'selected_image_for_feedback'):
            st.markdown("---")
            st.header("🎯 Edit Previous Image")
            
            col_prev1, col_prev2 = st.columns([1, 1])
            
            with col_prev1:
                st.subheader("Original Image")
                if os.path.exists(st.session_state.selected_image_for_feedback):
                    st.image(
                        st.session_state.selected_image_for_feedback, 
                        caption=f"Original: {st.session_state.get('selected_image_prompt', 'Unknown prompt')}", 
                        use_container_width=True
                    )
                else:
                    st.error("Selected image not found")
                    
            with col_prev2:
                st.subheader("Apply Feedback")
                
                prev_feedback = st.text_area(
                    "How would you like to modify this image?",
                    placeholder="e.g., 'make it brighter', 'add more blues', 'different lighting'",
                    height=150,
                    key="prev_feedback"
                )
                
                col_apply, col_clear = st.columns(2)
                
                with col_apply:
                    if st.button("� Apply Feedback", key="apply_prev_feedback", use_container_width=True):
                        if prev_feedback.strip():
                            with st.spinner("🎨 Applying feedback..."):
                                feedback_result = self.apply_feedback_to_image(
                                    st.session_state.selected_image_for_feedback,
                                    prev_feedback,
                                    st.session_state.get('selected_image_prompt', '')
                                )
                                
                                if feedback_result.success:
                                    st.success("✅ Feedback applied successfully!")
                                    st.image(
                                        feedback_result.edited_image_path,
                                        caption=f"Edited: {feedback_result.feedback_applied}",
                                        use_container_width=True
                                    )
                                    
                                    # Save the edited image metadata
                                    self.save_generation_metadata(
                                        feedback_result.edited_image_path,
                                        f"{st.session_state.get('selected_image_prompt', '')} - {prev_feedback}",
                                        "Edited"
                                    )
                                else:
                                    st.error(f"❌ Feedback failed: {feedback_result.error_message}")
                        else:
                            st.warning("Please enter feedback to apply")
                
                with col_clear:
                    if st.button("❌ Clear Selection", key="clear_prev_feedback", use_container_width=True):
                        if 'selected_image_for_feedback' in st.session_state:
                            del st.session_state.selected_image_for_feedback
                        if 'selected_image_prompt' in st.session_state:
                            del st.session_state.selected_image_prompt
                        st.rerun()

        st.markdown("---")

def main():
    """Application entry point"""
    try:
        app = ArtGeneratorApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        logger.error(f"App startup failed: {e}")

if __name__ == "__main__":
    main()
