from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os, json, uuid, io, re, sys
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import requests
from datetime import datetime
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

class FileTypes:
    PDF = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3


# ============================================================================
# CONFIGURATION
# ============================================================================
# Required for the protected GNP upload endpoints:
#   GNP_JWT_TOKEN=<JWT token>
#
# Optional:
#   GNP_FILE_API_URL=http://localhost:5034/api/v1/ingestion/
#   GNP_OCR_API_URL=http://localhost:5034/api/v1/ingestion/ocr
#   GNP_MASTER_DOCUMENT_BUCKET=gnp-master-documents
#   GNP_THUMBNAIL_BUCKET=gnp-thumbnails
# ============================================================================

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def auth_drive():
    """Authenticate with Google Drive using service account"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            'service_account.json',
            scopes=['https://www.googleapis.com/auth/drive']
        )   
        return creds
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise

def auth_drive_oauth():
    """Alternative OAuth authentication method"""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds



def init_cloudinary():
    cloudinary.config(
        cloud_name="pkdbucv7",
        api_key="658397586963537",
        api_secret="fcBKkncCorXWaT7RD2j2gjmTUrE",
        secure=True,
    )




def upload_thumbnail_to_cloudinary(thumbnail_buffer, asset_id, page_num=1):
    if thumbnail_buffer is None:
        logger.error("Thumbnail buffer is None")
        return None
    try:
        thumbnail_buffer.seek(0, os.SEEK_END)
        size = thumbnail_buffer.tell()
        thumbnail_buffer.seek(0)
        if size == 0:
            logger.error("Thumbnail buffer is empty")
            return None

        result = cloudinary.uploader.upload(
            thumbnail_buffer,
            folder="archive-thumbnails",
            public_id=f"thumb_{asset_id}_p{page_num}",
            resource_type="image",
            format="jpg",
            overwrite=True,
            invalidate=True,
        )
        url = result.get("secure_url")
        logger.info(f"Thumbnail uploaded: {url}")
        return url
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {e}")
        return None



# ============================================================================
# DATA CLASSES FOR CLASSIFICATION
# ============================================================================

@dataclass
class ExtractedContent:
    """Structured data extracted from OCR content"""
    categories: List[str]
    tags: List[str]
    entities: Dict[str, Any]
    summary: str
    content_type: str

# ============================================================================
# CONTENT CLASSIFIER
# ============================================================================

import json
import re
from typing import Dict, Any, Optional

class ContentClassifier:
    """Handles LLM-powered content classification using Qwen3 8B via Ollama"""
    
    def __init__(self, model_name="llama3.2", use_local_ollama=True):
        self.model_name = model_name
        self.use_local_ollama = use_local_ollama
        
    def classify_text(self, text: str, max_length: int = 9999) -> ExtractedContent:
        """Classify text content using Qwen3 8B via Ollama with robust JSON parsing"""
        

        #print("text presented to ContentClassifier ->")
        #print(text)

        truncated_text = text[:max_length] if len(text) > max_length else text
        
        # Improved prompt that asks for cleaner output
        prompt = f"""Analyze the following newspaper text and extract information. 
        Return ONLY valid JSON with no additional text, no markdown, no explanation.

        Text: {truncated_text}

        Required JSON format:
        {{"categories": ["category1"], "tags": ["tag1"], "content_type": "type", "summary": "summary"}}

        Categories can be: vacancy, tender, obituary, news, advertisement, announcement, editorial, sports, entertainment, business, politics
        Content type should be one or more of: vacancy, tender, obituary, news, advert

        Return ONLY the JSON object, nothing else."""
        
        try:
            if self.use_local_ollama:
                import ollama
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'You are a JSON-only document classifier. Never include explanations or markdown. Only output valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ])
                
                result = response['message']['content']

                print("llama result ->")
                print(result)
                
                # More robust JSON extraction
                parsed_data = self._parse_llm_json(result)
                
                return ExtractedContent(
                    categories=parsed_data.get('categories', []),
                    tags=parsed_data.get('tags', []),
                    entities={},
                    summary=parsed_data.get('summary', text),
                    content_type=parsed_data.get('content_type', 'news')
                )
            else:
                return self._keyword_classification(text)
                
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._keyword_classification(text)
    
    def _parse_llm_json(self, llm_response: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM response that might have formatting issues"""
        
        # Remove markdown code blocks
        llm_response = re.sub(r'```json\s*', '', llm_response)
        llm_response = re.sub(r'```\s*', '', llm_response)
        
        # Remove any leading/trailing whitespace
        llm_response = llm_response.strip()
        
        # Try to find JSON object in the response (between { and })
        json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
        if json_match:
            llm_response = json_match.group(0)
        
        # Fix common JSON syntax errors
        # Remove trailing commas before closing braces/brackets
        llm_response = re.sub(r',\s*}', '}', llm_response)
        llm_response = re.sub(r',\s*]', ']', llm_response)
        
        # Fix unquoted keys
        llm_response = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', llm_response)
        
        # Fix string values that might contain unescaped quotes
        def fix_string_values(match):
            key = match.group(1)
            value = match.group(2)
            # Escape quotes inside the string
            value = value.replace('"', '\\"')
            # Remove newlines
            value = value.replace('\n', ' ')
            return f'{key}"{value}"'
        
        # Find key-value pairs where value might be problematic
        pattern = r'("[a-zA-Z_]+"):\s*"([^"]*(?:\\"[^"]*)*)"'
        llm_response = re.sub(pattern, fix_string_values, llm_response)
        
        # Try multiple parsing attempts
        for attempt in range(3):
            try:
                return json.loads(llm_response)
            except json.JSONDecodeError as e:
                if attempt == 0:
                    # First attempt failed - try to fix common issues
                    # Replace single quotes with double quotes
                    llm_response = llm_response.replace("'", '"')
                elif attempt == 1:
                    # Extract anything that looks like a key-value pair
                    return self._extract_key_value_pairs(llm_response)
                else:
                    # Return default structure
                    logger.warning(f"Failed to parse JSON after multiple attempts: {e}")
                    return {
                        'categories': ['news'],
                        'tags': ['unclassified'],
                        'content_type': 'news',
                        'summary': 'Unable to classify content'
                    }
        
        return self._extract_key_value_pairs(llm_response)
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """Fallback method to extract key-value pairs using regex"""
        result = {
            'categories': ['news'],
            'tags': [],
            'content_type': 'news',
            'summary': ''
        }
        
        # Try to extract categories
        cat_match = re.search(r'"categories"\s*:\s*\[(.*?)\]', text, re.IGNORECASE)
        if cat_match:
            categories = re.findall(r'"([^"]+)"', cat_match.group(1))
            result['categories'] = categories if categories else ['news']
        
        # Try to extract content type
        type_match = re.search(r'"content_type"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if type_match:
            result['content_type'] = type_match.group(1)
        
        # Try to extract tags
        tags_match = re.search(r'"tags"\s*:\s*\[(.*?)\]', text, re.IGNORECASE)
        if tags_match:
            tags = re.findall(r'"([^"]+)"', tags_match.group(1))
            result['tags'] = tags
        
        # Try to extract summary
        summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if summary_match:
            result['summary'] = summary_match.group(1)[:200]
        
        return result
    
    def _keyword_classification(self, text: str) -> ExtractedContent:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        categories = []
        tags = []
        content_type = "news"
        
        # Define keyword mappings
        vacancy_keywords = ['vacancy', 'job', 'recruitment', 'employment', 'career', 'position', 'hiring', 'wanted']
        tender_keywords = ['tender', 'bid', 'procurement', 'contract', 'supply', 'quotation', 'invitation to tender']
        obituary_keywords = ['obituary', 'died', 'death', 'passed away', 'funeral', 'burial', 'in memoriam', 'late']
        
        # Check for content types
        vacancy_score = sum(1 for kw in vacancy_keywords if kw in text_lower)
        tender_score = sum(1 for kw in tender_keywords if kw in text_lower)
        obituary_score = sum(1 for kw in obituary_keywords if kw in text_lower)
        
        if vacancy_score > 0:
            categories.append('vacancy')
            tags.append('job-vacancy')
            content_type = 'vacancy'
        if tender_score > 0:
            categories.append('tender')
            tags.append('government-tender')
            content_type = 'tender'
        if obituary_score > 0:
            categories.append('obituary')
            tags.append('death-announcement')
            content_type = 'obituary'
        
        if not categories:
            categories.append('news')
            tags.append('general-news')
        
        # Extract potential job titles, tender numbers, etc.
        if content_type == 'vacancy':
            # Try to extract job titles using patterns
            job_matches = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:vacancy|position|job)', text)
            if job_matches:
                tags.append(f"job-{job_matches[0].lower().replace(' ', '-')[:30]}")
        
        return ExtractedContent(
            categories=categories,
            tags=tags,
            entities={},
            summary=self._extract_summary(text),
            content_type=content_type
        )
    
    def _extract_summary(self, text: str, max_length: int = 200) -> str:
        """Extract a meaningful summary from text"""
        # Try to get first sentence or paragraph
        sentences = re.split(r'[.!?]+', text)
        if sentences and len(sentences[0]) > 20:
            summary = sentences[0].strip()
        else:
            # Get first 200 characters
            summary = text[:max_length]
        
        # Clean up
        summary = re.sub(r'\s+', ' ', summary)
        return summary.strip()



# ============================================================================
# ADVANCED LAYOUT ANALYZER
# ============================================================================

class AdvancedLayoutAnalyzer:
    """Handles advanced layout analysis for multi-column documents"""
    
    def __init__(self, use_docling=False):  # Set to False by default to avoid dependency
        self.use_docling = use_docling
        
    def extract_text_with_layout(self, pdf_path: str, page_number: int = None) -> str:
        """Extract text with proper layout handling"""
        if self.use_docling:
            try:
                return self._extract_with_docling(pdf_path, page_number)
            except Exception as e:
                logger.warning(f"Docling failed: {e}, using PyMuPDF")
                return self._extract_with_pymupdf_layout(pdf_path, page_number)
        else:
            return self._extract_with_pymupdf_layout(pdf_path, page_number)
    
    def _extract_with_docling(self, pdf_path: str, page_number: int = None) -> str:
        """Use Docling for advanced layout analysis"""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            
            if page_number is not None:
                if page_number - 1 < len(result.document.pages):
                    page_text = result.document.pages[page_number - 1].text
                    return self._clean_extracted_text(page_text)
                return ""
            
            full_text = ""
            for page in result.document.pages:
                full_text += page.text + "\n\n"
            
            return self._clean_extracted_text(full_text)
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            return ""
    
    def _extract_with_pymupdf_layout(self, pdf_path: str, page_number: int = None) -> str:
        """Use PyMuPDF with layout analysis"""
        try:
            doc = fitz.open(pdf_path)
            
            if page_number is not None:
                pages = [doc[page_number - 1]]
            else:
                pages = doc
            
            extracted_text = []
            for page in pages:
                text = page.get_text("text", sort=True)
                extracted_text.append(text)
            
            doc.close()
            return self._clean_extracted_text("\n\n".join(extracted_text))
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text

# ============================================================================
# ENHANCED OCR PIPELINE
# ============================================================================

class EnhancedOCRPipeline:
    """Main OCR pipeline with advanced classification"""
    
    def __init__(self, classifier_model="llama3.2", use_docling=False):
        self.classifier = ContentClassifier(model_name=classifier_model)
        self.layout_analyzer = AdvancedLayoutAnalyzer(use_docling=use_docling)
        
    def process_ocr_text(self, text: str, file_name: str = "") -> Dict[str, Any]:
        """Process OCR text with classification"""
        cleaned_text = self._clean_ocr_text(text)
        classification = self.classifier.classify_text(cleaned_text)
        
        additional_tags = self._extract_date_tags(file_name)
        all_tags = list(set(classification.tags + additional_tags))
        
        return {
            'original_text': cleaned_text,
            'categories': classification.categories,
            'tags': all_tags,
            'content_type': classification.content_type,
            'entities': classification.entities,
            'summary': classification.summary
        }
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text"""
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        return text.strip()
    
    def _extract_date_tags(self, file_name: str) -> List[str]:
        """Extract date information from filename"""
        tags = []
        date_match = re.search(r'(\d{4})', file_name)
        if date_match:
            tags.append(f"year-{date_match.group(1)}")
        return tags

# ============================================================================
# OBFUSCATION FUNCTIONS
# ============================================================================

def obfuscate_thumbnail(img, method="balanced", output_size=(300, 300)):
    """Apply irreversible obfuscation to thumbnails and return as BytesIO buffer"""
    try:
        # Create a copy to avoid modifying original
        img = img.copy()
        
        # Resize
        img.thumbnail(output_size, Image.Resampling.LANCZOS)
        
        if method == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=10))
        elif method == "pixelate":
            small = img.resize((40, 40), Image.Resampling.NEAREST)
            img = small.resize(img.size, Image.Resampling.NEAREST)
        elif method == "balanced":
            # Mild blur
            img = img.filter(ImageFilter.GaussianBlur(radius=0.05))
            # Gentle pixelation
            small = img.resize((120, 120), Image.Resampling.BILINEAR)
            img = small.resize(output_size, Image.Resampling.NEAREST)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Add light noise
            np_img = np.array(img)
            noise = np.random.randint(0, 20, np_img.shape, dtype=np.uint8)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(np_img)
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to buffer with compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)
        
        # Verify buffer has content
        buffer_size = buffer.getbuffer().nbytes
        logger.info(f"Created thumbnail buffer of size: {buffer_size} bytes")
        
        if buffer_size == 0:
            logger.error("Created buffer is empty!")
            return None
            
        return buffer
        
    except Exception as e:
        logger.error(f"Error in obfuscate_thumbnail: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_obfuscated_thumbnail_from_image(original_image_path, method="balanced"):
    """Create obfuscated thumbnail from image file"""
    try:
        if not os.path.exists(original_image_path):
            logger.error(f"Image file does not exist: {original_image_path}")
            return None
            
        logger.info(f"Opening image: {original_image_path}")
        with Image.open(original_image_path) as img:
            logger.info(f"Image opened successfully. Size: {img.size}, Mode: {img.mode}")
            buffer = obfuscate_thumbnail(img, method=method)
            return buffer
    except Exception as e:
        logger.error(f"Error creating thumbnail from image: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_obfuscated_thumbnail_from_pdf_page(page, method="balanced", output_size=(400, 400)):
    """Create obfuscated thumbnail from PDF page"""
    try:
        logger.info("Converting PDF page to image")
        
        # Convert PDF page to image
        zoom = 1.5
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        logger.info(f"PDF page converted. Image data size: {len(img_data)} bytes")
        
        img = Image.open(io.BytesIO(img_data))
        logger.info(f"PIL Image created. Size: {img.size}, Mode: {img.mode}")
        
        # Apply obfuscation
        buffer = obfuscate_thumbnail(img, method=method, output_size=output_size)
        
        return buffer
        
    except Exception as e:
        logger.error(f"Error creating thumbnail from PDF page: {e}")
        import traceback
        traceback.print_exc()
        return None



def obfuscate_pdf_thumbnail_smart(page, output_size=(400, 400)):
    """
    Smart obfuscation that:
    1. Preserves document layout and visual appeal
    2. Makes text unreadable through character replacement
    3. Maintains image quality for non-text elements
    """
    try:
        # Convert PDF page to high-quality image
        zoom = 2.0  # Higher quality for better text detection
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        
        # Open with PIL
        img = Image.open(io.BytesIO(img_data))
        
        # Resize to thumbnail size
        img.thumbnail(output_size, Image.Resampling.LANCZOS)
        
        # Try to detect and obfuscate text
        try:
            import pytesseract
            from PIL import ImageDraw
            
            # Get text regions
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Create obfuscated version
            img_result = img.copy()
            draw = ImageDraw.Draw(img_result)
            
            # Obfuscate each text region
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 40:  # Only confident detections
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Skip very small regions (likely not text)
                    if w < 10 or h < 10:
                        continue
                    
                    # Method: Replace text with noise pattern that matches background
                    # Extract the background around text
                    margin = 2
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(img.width, x + w + margin)
                    y2 = min(img.height, y + h + margin)
                    
                    # Create a noise pattern that blends with background
                    background = np.array(img.crop((x1, y1, x2, y2)))
                    
                    # Generate similar noise pattern
                    noise = np.random.normal(0, 20, background.shape)
                    noisy_bg = np.clip(background + noise, 0, 250).astype(np.uint8)
                    
                    # Paste noise over text area
                    img_result.paste(Image.fromarray(noisy_bg), (x1, y1))
                    
                    # Add subtle texture lines to ensure text is unreadable
                    for _ in range(5):
                        line_x = x1 + np.random.randint(0, x2-x1)
                        line_y = y1 + np.random.randint(0, y2-y1)
                        draw.line([(line_x, line_y), (line_x+5, line_y+5)], 
                                 fill=(128,128,128,150), width=1)
            
            # Final enhancement: slight overall sharpening to maintain beauty
            img_result = img_result.filter(ImageFilter.UnsharpMask(radius=0.3, percent=30, threshold=0))
            
            # Convert to buffer
            buffer = io.BytesIO()
            img_result.save(buffer, format='JPEG', quality=90, optimize=True)
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            print(f"Smart text detection failed: {e}, using alternative method")
            # Fallback to gentle text noise
            img_array = np.array(img)
            # Add subtle noise only to high-frequency areas (text)
            from scipy import ndimage
            edges = np.abs(ndimage.sobel(np.array(img.convert('L')).astype(float)))
            text_mask = edges > np.percentile(edges, 80)
            
            noise = np.random.normal(0, 40, img_array.shape)
            for c in range(3):
                img_array[:,:,c] = np.where(text_mask, 
                                           np.clip(img_array[:,:,c] + noise[:,:,c], 0, 255),
                                           img_array[:,:,c])
            
            result = Image.fromarray(img_array.astype(np.uint8))
            buffer = io.BytesIO()
            result.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            return buffer
            
    except Exception as e:
        print(f"Error in smart obfuscation: {e}")
        return None



def upload_file_asset(file_content, asset_id: str, bucket_name: str, file_name: str,
                     content_type: str = "application/octet-stream"):
    """
    Upload a file asset to the GNP file service.

    Equivalent to the TypeScript uploadFileAsset() method:
      POST g3/upload-file/{asset_id}
      multipart fields: bucketName, file
      Authorization: Bearer <JWT>

    `file_content` may be a BytesIO object, bytes, or a filesystem path.
    """
    if not asset_id:
        raise ValueError("asset_id is required")

    url = f"https://api.graphicnewsplus.com/api/v1/g3/upload-file/{asset_id}"

    headers = {
        "Authorization": f"Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Inlhd0BnbWFpbC5jb20iLCJleHAiOjIxMDQyNDQyMzAsImlhdCI6MTc4ODAyMDIzMCwiaXNzIjoiR25wIiwidXNlcklkIjoiYjAzZWYwZTEtMGM3My00NjA1LWEzMzEtMWM0YjkyYTUwYzExIiwidXNlcm5hbWUiOiJ5YXd0b2cifQ.ypcEtUGiliTD_-RueOxkNxaMcZ_BV2-OuRcMPqa-YsI"
    }

    # Track if we opened a file handle ourselves
    file_handle = None
    close_file = False
    temp_file_path = None

    try:
        # Requests needs a file-like object for multipart upload.
        if isinstance(file_content, (str, os.PathLike)):
            temp_file_path = str(file_content)
            file_handle = open(file_content, "rb")
            close_file = True
        elif isinstance(file_content, bytes):
            file_handle = io.BytesIO(file_content)
        else:
            file_handle = file_content
            file_handle.seek(0)

        files = {
            "file": (file_name, file_handle, content_type)
        }
        data = {
            "bucketName": bucket_name
        }

        logger.info(
            f"Uploading asset {asset_id} to {url} "
            f"(bucket={bucket_name}, file={file_name})"
        )

        response = requests.post(
            url,
            headers=headers,
            data=data,
            files=files,
            timeout=300
        )

        if close_file and file_handle:
            file_handle.close()

        if response.ok:
            try:
                result = response.json()
            except ValueError:
                result = {"status": "success", "fileName": file_name}

            logger.info(
                f"Asset uploaded successfully: {asset_id} "
                f"(status={response.status_code})"
            )
            
            # Clean up temporary file after successful upload
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
            
            return result

        logger.error(
            f"Asset upload failed: status={response.status_code}, "
            f"response={response.text}"
        )
        
        # Clean up temporary file even if upload failed
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file after failed upload: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
        
        return None

    except Exception as e:
        logger.error(f"Error uploading asset {asset_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up temporary file on exception
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file after error: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")
        
        return None




def upload_obfuscated_thumbnail(thumbnail_buffer, asset_id: str, page_num=1):
    """Upload an obfuscated thumbnail using the GNP file asset API."""
    if thumbnail_buffer is None:
        logger.error("Thumbnail buffer is None, cannot upload")
        return None

    try:
        thumbnail_buffer.seek(0, os.SEEK_END)
        buffer_size = thumbnail_buffer.tell()
        thumbnail_buffer.seek(0)

        if buffer_size == 0:
            logger.error("Thumbnail buffer is empty, cannot upload")
            return None

        thumbnail_bucket = os.getenv(
            "GNP_THUMBNAIL_BUCKET",
            "gnp-thumbnails"
        )
        thumbnail_filename = f"thumb_{asset_id}_p{page_num}.jpg"

        return upload_file_asset(
            file_content=thumbnail_buffer,
            asset_id=asset_id,
            bucket_name=thumbnail_bucket,
            file_name=thumbnail_filename,
            content_type="image/jpeg"
        )

    except Exception as e:
        logger.error(f"Error uploading thumbnail asset: {e}")
        return None


# Add this debug function to test thumbnail creation
def test_thumbnail_creation():
    """Test function to verify thumbnail creation works"""
    try:
        # Create a test image
        test_img = Image.new('RGB', (800, 600), color='white')
        
        # Draw some text on it for testing
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.text((50, 50), "Test OCR Content", fill='black')
        
        # Create thumbnail
        buffer = obfuscate_thumbnail(test_img, method="balanced")
        
        if buffer:
            # Save to file for inspection
            with open('test_thumbnail.jpg', 'wb') as f:
                f.write(buffer.getbuffer())
            logger.info("Test thumbnail saved to test_thumbnail.jpg")
            return True
        else:
            logger.error("Failed to create test thumbnail")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

# ============================================================================
# PDF PROCESSING FUNCTIONS
# ============================================================================

def upload_pdf_page(file_path: str, asset_id: str, page_number: int):
    """
    Create a single-page PDF from the source file and upload it as the
    master document using the GNP file asset API.
    """
    temp_pdf_path = None
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_number - 1)

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)

        temp_pdf_path = os.path.join(
            tempfile.gettempdir(),
            f"{asset_id}_page_{page_number}.pdf"
        )
        new_doc.save(temp_pdf_path)
        new_doc.close()
        doc.close()

        master_bucket = os.getenv(
            "GNP_MASTER_DOCUMENT_BUCKET",
            "gnp-master-documents"
        )
        filename = f"{asset_id}_page_{page_number}.pdf"

        return upload_file_asset(
            file_content=temp_pdf_path,
            asset_id=asset_id,
            bucket_name=master_bucket,
            file_name=filename,
            content_type="application/pdf"
        )

    except Exception as e:
        logger.error(f"Error uploading PDF page {page_number}: {e}")
        return None
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


# ============================================================================
# API FUNCTIONS
# ============================================================================

def create_enhanced_publication(
    publication_date,
    page_number,
    classification_result,
    asset_id,
    file_type,
    edition_number,
    thumbnail_url=None,
):
    """Create an enhanced publication using the shared asset_id."""
    publication_text = classification_result['original_text'].replace('\n', ' ')

    # Log the thumbnail_url
    logger.info(f"=== thumbnail_url DEBUG ===")
    logger.info(f"thumbnail_url: {thumbnail_url}")

    # Log the initial file type
    logger.info(f"=== FILE TYPE CONVERSION DEBUG ===")
    logger.info(f"Original file_type: {file_type} (type: {type(file_type)})")

    # Resolve file type to the API FileTypes enum before the request.
    file_type_map = {
        "pdf": 0,
        "image": 1,
        "video": 2,
        "audio": 3,
    }
    
    # Convert file_type to integer if it's a string
    if isinstance(file_type, str):
        normalized_file_type = file_type.strip().lower()
        logger.info(f"Normalized file_type string: '{normalized_file_type}'")
        
        if normalized_file_type not in file_type_map:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
        
        converted_file_type = file_type_map[normalized_file_type]
        logger.info(f"Converted to integer: {converted_file_type} (type: {type(converted_file_type)})")
        logger.info(f"Mapping: '{normalized_file_type}' -> {converted_file_type}")
        
        file_type = converted_file_type  # Assign the converted value back
        
    elif isinstance(file_type, int):
        logger.info(f"file_type is already an integer: {file_type}")
        if file_type not in (FileTypes.PDF, FileTypes.IMAGE, FileTypes.VIDEO, FileTypes.AUDIO):
            logger.error(f"Unsupported FileTypes value: {file_type}")
            raise ValueError(f"Unsupported FileTypes value: {file_type}")
    else:
        logger.error(f"Invalid file_type type: {type(file_type)}")
        raise ValueError(f"Invalid file_type type: {type(file_type)}")

    # Log the final file type
    logger.info(f"Final file_type: {file_type} (type: {type(file_type)})")
    logger.info(f"=== END FILE TYPE CONVERSION ===")

    url = "https://api.graphicnewsplus.com/api/v1/ingestion/ocr"

    if '-' in str(page_number):
        page_number = str(page_number).split('-')[0]

    data = {
        "id": asset_id,
        "title": classification_result['summary'][:100]
            if classification_result['summary']
            else publication_text[:50],
        "publicationDate": publication_date,
        "fileType": file_type,
        "tags": classification_result['tags'],
        "categories": classification_result['categories'],
        "contentType": classification_result['content_type'],
        "pageSummary": classification_result['summary'],
        "datePublished": publication_date,
        "pageText": publication_text,
        "pageNumber": int(page_number),
        "editionNumber": edition_number,
        "thumbnailUrl": thumbnail_url,
    }

    logger.info("OCR ingestion payload:")
    logger.info(f"  - fileType: {data['fileType']} (type: {type(data['fileType'])})")
    logger.info(f"  - id: {data['id']}")
    logger.info(f"  - contentType: {data['contentType']}")
    logger.info(f"  - pageNumber: {data['pageNumber']}")

    # Keep authentication consistent with the protected asset endpoints.
    headers = {"Content-Type": "application/json"}

    try:
        logger.info(f"Sending POST request to {url}")
        response = requests.post(
            url,
            json=data,
            timeout=300
        )

        if response.ok:
            logger.info(
                f'Newspaper page ingested: '
                f'{classification_result["content_type"]}'
            )
            return response.json()

        logger.error(
            f"Failed to create enhanced publication: "
            f"{response.status_code} - {response.text}"
        )
        return None

    except Exception as e:
        logger.error(f"Error creating enhanced publication: {e}")
        return None


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def extract_info(filename):
    """Extract date and page info from filename"""
    pattern = r'(\d{4})(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)(0?\d{1,2}).*?PAGE(-?\d+(?:[-&]\d+)?)'
    
    match = re.search(pattern, str(filename).upper(), re.IGNORECASE)
    
    if match:
        year, month_str, day, page_range = match.groups()
        
        # Normalize month string to a recognizable three-letter abbreviation format for strptime
        month_abbr = month_str[:3].title()
        
        # Convert month abbreviation to a number
        month_number = datetime.strptime(month_abbr, '%b').month
        
        # Remove any leading zeros from day
        day = str(int(day))
        
        date = f"{year}-{month_number:02d}-{day}"
        
        # Ensure page range format is consistent
        page_range = page_range.replace('&', '-')

        # if page range starts with a hyphen, remove it
        if page_range.startswith('-'):
            page_range = page_range[1:]
        
        return date, page_range
    else:
        return None, None


def extract_edition_number(text: str) -> Optional[str]:
    """
    Extract edition number from text using patterns like:
    - No. 22148
    - No 22148
    - NO.22148
    """
    if not text:
        return None

    # Regex explanation:
    # \bNo\.?     -> matches 'No' or 'No.'
    # \s*         -> optional spaces
    # (\d{3,})    -> capture 3+ digit number (edition numbers are usually large)
    pattern = r'\bNo\.?\s*(\d{3,})\b'

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def extract_publication_date(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        # Standard format
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'(\d{1,2}),\s*(\d{4})',

        # Missing comma after weekday
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'(\d{1,2})\s*(\d{4})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parts = match.groups()

            # Handle both pattern variants
            if len(parts) == 3:
                month, day, year = parts
            elif len(parts) == 4:
                _, month, day, year = parts

            try:
                parsed_date = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
                return parsed_date.strftime("%Y-%m-%d")
            except Exception:
                return f"{month} {day}, {year}"

    return None


def download_file(service, file_id, file_name):
    """Download file from Google Drive"""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    
    with open(file_name, 'wb') as f:
        fh.seek(0)
        f.write(fh.read())
    logger.info(f"Downloaded {file_name}")
    
    file_path = os.path.join('.', file_name)
    _, extension = os.path.splitext(file_path)
    
    if extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        ocr_image(file_path, file_id, service)
    elif extension.lower() == '.pdf':
        process_pdf(file_path, file_id, service)

def ocr_image(file_path, file_id, service):
    """Process image with OCR, upload assets, and create publication."""
    try:
        logger.info(f"Processing image: {file_path}")
        pipeline = EnhancedOCRPipeline(use_docling=False)

        img = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(img)

        classification_result = pipeline.process_ocr_text(
            extracted_text, file_path
        )
        logger.info(
            f"Image classified as: "
            f"{classification_result['content_type']}"
        )

        date, page = extract_info(file_path)

        if not (date and page):
            logger.warning(
                "Could not extract date and page number from filename. "
                "Skipping..."
            )
            return

        logger.info(f"Date: {date}, Page: {page}")

        # One UUID identifies the logical asset/publication. The same ID is
        # used for both the thumbnail upload and the master document upload.
        asset_id = str(uuid.uuid4())
        logger.info(f"Generated asset_id: {asset_id}")

        # 1. Upload obfuscated thumbnail.
        thumbnail_buffer = create_obfuscated_thumbnail_from_image(
            file_path, method="balanced"
        )
        thumbnail_result = None
        if thumbnail_buffer:
            thumbnail_url = upload_thumbnail_to_cloudinary(
                thumbnail_buffer, asset_id, page_num=1
            )

        if not thumbnail_url:
            logger.warning("Thumbnail upload failed")

        # 2. Upload original image as the master document.
        master_bucket = os.getenv(
            "GNP_MASTER_DOCUMENT_BUCKET",
            "gnp-master-documents"
        )
        master_result = upload_file_asset(
            file_content=file_path,
            asset_id=asset_id,
            bucket_name=master_bucket,
            file_name=os.path.basename(file_path),
            content_type="image/jpeg"
            if file_path.lower().endswith((".jpg", ".jpeg"))
            else "image/png"
        )

        if not master_result:
            logger.error("Master document upload failed; skipping publication")
            return

        # 3. Create publication using asset_id instead of thumbnail_id/document_id.
        create_enhanced_publication(
            date,
            page,
            classification_result,
            asset_id,
            "image",
            "N/A",
            thumbnail_url=thumbnail_url,
        )
        logger.info(
            f"Publication created successfully for asset_id={asset_id}"
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()



def process_pdf(file_path, file_id, service):
    """Process PDF, upload obfuscated thumbnails and master documents."""
    try:
        pipeline = EnhancedOCRPipeline(use_docling=False)
        doc = fitz.open(file_path)

        total_pages = len(doc)
        base_filename = os.path.basename(file_path)

        edition_number = ""
        publication_date = ""

        logger.info(
            f"Processing PDF with {total_pages} pages: {base_filename}"
        )

        for page_number in range(total_pages):
            logger.info(f"\\n{'=' * 50}")
            logger.info(
                f"Processing page {page_number + 1} of {total_pages}"
            )
            logger.info(f"{'=' * 50}")

            page = doc.load_page(page_number)

            page_text = pipeline.layout_analyzer.extract_text_with_layout(
                file_path, page_number + 1
            )

            if page_number == 0:
                edition_number = extract_edition_number(page_text) or ""
                publication_date = extract_publication_date(page_text) or ""

                logger.info(
                    f"Extracted edition number: {edition_number}"
                )
                logger.info(
                    f"Extracted publication date: {publication_date}"
                )

            if not page_text or len(page_text.strip()) < 10:
                page_text = page.get_text("text")

            logger.info(
                f"Extracted text length: {len(page_text)} characters"
            )

            classification_result = pipeline.process_ocr_text(
                page_text, base_filename
            )
            logger.info(
                f"Page {page_number + 1} classified as: "
                f"{classification_result['content_type']}"
            )

            # Generate ONE asset_id for this page. The same asset_id is used
            # by both g3 upload endpoints and in create_enhanced_publication.
            asset_id = str(uuid.uuid4())
            logger.info(
                f"Generated asset_id for page {page_number + 1}: {asset_id}"
            )

            # 1. Create and upload obfuscated thumbnail.
            thumbnail_buffer = obfuscate_pdf_thumbnail_smart(
                page, output_size=(500, 333)
            )

            thumbnail_result = None
            if thumbnail_buffer:
                thumbnail_url = upload_thumbnail_to_cloudinary(
                    thumbnail_buffer,
                    asset_id,
                    page_num=page_number + 1,
                )

                if thumbnail_url:
                    logger.info(f"Thumbnail uploaded: {thumbnail_url}")
                else:
                    logger.error(
                        f"Thumbnail upload failed for page "
                        f"{page_number + 1}"
                    )
            else:
                logger.error(
                    f"Failed to create thumbnail for page "
                    f"{page_number + 1}"
                )

            # 2. Create and upload the single-page master document using
            # the SAME asset_id.
            logger.info(
                f"Uploading master document for page {page_number + 1}"
            )
            master_result = upload_pdf_page(
                file_path,
                asset_id,
                page_number + 1
            )

            if not master_result:
                logger.error(
                    f"Master document upload failed for page "
                    f"{page_number + 1}; skipping publication"
                )
                continue

            pub_date = (
                publication_date
                if publication_date
                else datetime.now().strftime("%Y-%m-%d")
            )

            # 3. Create publication using only asset_id.
            logger.info(
                f"Creating publication for page {page_number + 1} "
                f"with asset_id={asset_id}"
            )

            create_enhanced_publication(
                pub_date,
                str(page_number + 1),
                classification_result,
                asset_id,
                "pdf",
                edition_number,
                thumbnail_url=thumbnail_url
            )

            logger.info(
                f"Successfully processed page {page_number + 1}"
            )

        doc.close()

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary PDF file: {file_path}")

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        import traceback
        traceback.print_exc()



def traverse_and_download(service, folder_id, base_path=''):
    """Traverse Google Drive folders and download files"""
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        pageSize=100, 
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])
    
    for item in items:
        logger.info(f"Found {item['name']}")
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            new_path = os.path.join(base_path, item['name'])
            os.makedirs(new_path, exist_ok=True)
            traverse_and_download(service, item['id'], new_path)
        else:
            file_path = os.path.join(base_path, item['name'])
            download_file(service, item['id'], file_path)

def read_folder_ids(file_path):
    """Read folder IDs from file"""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def main():
    """Main entry point"""
    init_cloudinary()
    creds = auth_drive()
    service = build('drive', 'v3', credentials=creds)
    
    folder_ids = read_folder_ids('folder_ids.txt')
    logger.info(f"Processing folders: {folder_ids}")
    
    for folder_id in folder_ids:
        logger.info(f"Traversing folder: {folder_id}")
        traverse_and_download(service, folder_id)

if __name__ == '__main__':
    main()