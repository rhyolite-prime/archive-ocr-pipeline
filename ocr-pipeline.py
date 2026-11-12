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

import argparse
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONCURRENCY SUPPORT
# ============================================================================
# The pipeline is dominated by I/O: every page triggers an Ollama
# classification round-trip, a Cloudinary upload, a master-document upload and
# a publication POST.  Those are independent per page, so we run them on a
# thread pool.
#
# The pieces of shared state that are NOT safe to share across threads are
# handled explicitly below:
#
#   * Google Drive clients are built per thread.  ``googleapiclient`` uses
#     ``httplib2``, whose connections are not thread-safe, so each worker gets
#     its own transport while sharing one (read-mostly) credentials object.
#   * PyMuPDF/MuPDF objects are never shared between threads.  Each worker
#     opens its own ``fitz.Document``.  MuPDF also has some library-level
#     global state, so access is additionally guarded by ``_FITZ_LOCK``
#     (disable with OCR_FITZ_LOCK=0 if you know your build is safe).
#   * The Ollama client gets one instance per thread (see ContentClassifier).
#   * ``cloudinary.config()`` and the logger are configured once, up front,
#     and only read afterwards.
# ============================================================================

#: Number of logical CPUs available to this process.  On Apple Silicon this is
#: the total of performance + efficiency cores (12 on an M2 Max), not just the
#: performance ones.
CPU_COUNT = os.cpu_count() or 4

#: Default number of pages processed concurrently.  Overridable with
#: --page-workers or the OCR_PAGE_WORKERS environment variable.
#:
#: One worker per CPU is the right default because the per-page work is
#: largely CPU-bound (tesseract + PyMuPDF rasterising) once the network calls
#: are running in parallel.  Going far above the core count only adds context
#: switching - see configure_tesseract_env() for the other half of the story.
DEFAULT_PAGE_WORKERS = int(os.getenv("OCR_PAGE_WORKERS", min(16, CPU_COUNT)))

#: Default number of files downloaded+dispatched concurrently.  Downloads are
#: network-bound and cost almost no CPU, so they do not compete with OCR.
DEFAULT_FILE_WORKERS = int(os.getenv("OCR_FILE_WORKERS", 3))

#: Guard around MuPDF entry points.  See the note above.
_FITZ_LOCK = threading.RLock()
_FITZ_LOCK_ENABLED = os.getenv("OCR_FITZ_LOCK", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# Tesseract / OpenMP thread budget
# ---------------------------------------------------------------------------
# tesseract is spawned as a *subprocess* per OCR call, and the Homebrew/macOS
# builds are compiled with OpenMP: by default each process starts as many
# threads as there are cores.  Run 8 of those concurrently on an 8-core budget
# and you get 8 x 8 threads fighting over the same cores, which is measurably
# slower than serial because every thread pays for context switches.
#
# pytesseract passes ``os.environ`` straight to the subprocess, so setting
# OMP_THREAD_LIMIT here propagates to every tesseract run.
#
# Rule of thumb: page_workers x OMP_THREAD_LIMIT should approximate the number
# of cores you are willing to give the pipeline.
#
#   OMP_THREAD_LIMIT=1  -> one single-threaded tesseract per worker. Best when
#                          you run one worker per core (the default).
#   OMP_THREAD_LIMIT=2-4 -> fewer, fatter workers. Slightly better cache
#                          locality, at the cost of coarser scheduling.

def configure_tesseract_env(page_workers: Optional[int] = None,
                            omp_thread_limit: Optional[int] = None) -> int:
    """
    Pin tesseract's OpenMP thread count so concurrent workers do not
    oversubscribe the CPU, and export it for tesseract subprocesses.

    Returns the limit that is in effect.  An explicit OMP_THREAD_LIMIT in the
    environment always wins over the argument.
    """
    env_limit = os.getenv("OMP_THREAD_LIMIT")
    if env_limit not in (None, ""):
        try:
            limit = max(1, int(env_limit))
        except ValueError:
            logger.warning(f"Ignoring non-numeric OMP_THREAD_LIMIT={env_limit!r}")
            limit = max(1, int(omp_thread_limit or 1))
    else:
        limit = max(1, int(omp_thread_limit or 1))

    # pytesseract passes os.environ to the tesseract subprocess, so these
    # reach it.  OMP_NUM_THREADS is setdef so an explicit value is preserved.
    os.environ["OMP_THREAD_LIMIT"] = str(limit)
    os.environ.setdefault("OMP_NUM_THREADS", str(limit))
    return limit


class _fitz_guard:
    """Context manager serialising MuPDF calls when the lock is enabled."""

    __slots__ = ()

    def __enter__(self):
        if _FITZ_LOCK_ENABLED:
            _FITZ_LOCK.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        if _FITZ_LOCK_ENABLED:
            _FITZ_LOCK.release()
        return False


# ---------------------------------------------------------------------------
# Thread-safe progress counter
# ---------------------------------------------------------------------------

class ProgressCounter:
    """Thread-safe counter used to report 'x/y' progress from worker threads."""

    def __init__(self, total: int = 0, label: str = "pages"):
        self._lock = threading.Lock()
        self._done = 0
        self._failed = 0
        self._total = total
        self._label = label
        self._started = time.time()

    def set_total(self, total: int) -> None:
        with self._lock:
            self._total = total

    def add_total(self, extra: int) -> None:
        with self._lock:
            self._total += extra

    def tick(self, ok: bool = True) -> None:
        with self._lock:
            if ok:
                self._done += 1
            else:
                self._failed += 1
            done, failed, total = self._done, self._failed, self._total
            elapsed = time.time() - self._started

        if done + failed == total or (done + failed) % 10 == 0:
            rate = (done + failed) / elapsed if elapsed > 0 else 0
            logger.info(
                f"[progress] {done + failed}/{total} {self._label} "
                f"(ok={done}, failed={failed}, {rate:.2f}/s)"
            )

    @property
    def failed(self) -> int:
        with self._lock:
            return self._failed

    @property
    def done(self) -> int:
        with self._lock:
            return self._done


# ---------------------------------------------------------------------------
# Per-thread Google Drive client
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_drive_credentials: Optional[Any] = None
_drive_credentials_lock = threading.Lock()


def set_drive_credentials(credentials) -> None:
    """Register the credentials used by every worker thread's Drive client."""
    global _drive_credentials
    with _drive_credentials_lock:
        _drive_credentials = credentials


def get_drive_service():
    """
    Return a Drive client bound to the current thread.

    ``httplib2.Http`` (used underneath ``googleapiclient``) is not thread-safe,
    so every worker thread gets its own ``AuthorizedHttp`` transport.  The
    credentials object itself is shared; ``AuthorizedHttp`` only refreshes a
    token when it has actually expired.
    """
    service = getattr(_thread_local, "drive_service", None)
    if service is not None:
        return service

    with _drive_credentials_lock:
        creds = _drive_credentials

    if creds is None:
        # Called without main() having set credentials (e.g. from a REPL).
        creds = auth_drive()
        set_drive_credentials(creds)

    try:
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp

        authorized = AuthorizedHttp(creds, http=httplib2.Http(timeout=300))
        service = build("drive", "v3", http=authorized, cache_discovery=False)
    except ImportError:  # pragma: no cover - optional dependency layout
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

    _thread_local.drive_service = service
    return service

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

    #: One Ollama client per thread.  The client wraps an ``httpx.Client``,
    #: which is not safe to share between threads while streaming/reading.
    _client_tls = threading.local()

    def __init__(self, model_name="llama3.2", use_local_ollama=True):
        self.model_name = model_name
        self.use_local_ollama = use_local_ollama
        self._cache: Dict[str, ExtractedContent] = {}
        self._cache_lock = threading.Lock()

    def _get_ollama_client(self):
        client = getattr(ContentClassifier._client_tls, "client", None)
        if client is None:
            import ollama
            host = os.getenv("OLLAMA_HOST") or None
            timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
            client = ollama.Client(host=host, timeout=timeout) if host else ollama.Client(timeout=timeout)
            ContentClassifier._client_tls.client = client
        return client

    def classify_text(self, text: str, max_length: int = 9999) -> ExtractedContent:
        """Classify text content using Qwen3 8B via Ollama with robust JSON parsing"""

        #print("text presented to ContentClassifier ->")
        #print(text)

        truncated_text = text[:max_length] if len(text) > max_length else text

        # Identical pages are common in archives (blank/repeated pages) and the
        # LLM call dominates the runtime, so memoise on the prompt text.
        cache_key = truncated_text
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

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
                client = self._get_ollama_client()
                response = client.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'You are a JSON-only document classifier. Never include explanations or markdown. Only output valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ])

                result = response['message']['content']

                logger.debug("llama result -> %s", result)

                # More robust JSON extraction
                parsed_data = self._parse_llm_json(result)

                extracted = ExtractedContent(
                    categories=parsed_data.get('categories', []),
                    tags=parsed_data.get('tags', []),
                    entities={},
                    summary=parsed_data.get('summary', text),
                    content_type=parsed_data.get('content_type', 'news')
                )
                with self._cache_lock:
                    self._cache[cache_key] = extracted
                return extracted
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
        
    def extract_text_with_layout(self, pdf_path: str, page_number: int = None,
                                 doc: "fitz.Document" = None) -> str:
        """
        Extract text with proper layout handling.

        ``doc`` may be supplied by a caller that already has the PDF open (as
        the concurrent page workers do), in which case it is reused instead of
        opening the file again.  The document is *not* closed here.
        """
        if self.use_docling:
            try:
                return self._extract_with_docling(pdf_path, page_number)
            except Exception as e:
                logger.warning(f"Docling failed: {e}, using PyMuPDF")
                return self._extract_with_pymupdf_layout(pdf_path, page_number, doc=doc)
        else:
            return self._extract_with_pymupdf_layout(pdf_path, page_number, doc=doc)
    
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
    
    def _extract_with_pymupdf_layout(self, pdf_path: str, page_number: int = None,
                                     doc: "fitz.Document" = None) -> str:
        """Use PyMuPDF with layout analysis"""
        owned_doc = doc is None
        try:
            if owned_doc:
                with _fitz_guard():
                    doc = fitz.open(pdf_path)

            if page_number is not None:
                pages = [doc[page_number - 1]]
            else:
                pages = doc

            extracted_text = []
            with _fitz_guard():
                for page in pages:
                    text = page.get_text("text", sort=True)
                    extracted_text.append(text)

            if owned_doc:
                with _fitz_guard():
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
        with _fitz_guard():
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
        with _fitz_guard():
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

def upload_pdf_page(file_path: str, asset_id: str, page_number: int,
                    doc: "fitz.Document" = None):
    """
    Create a single-page PDF from the source file and upload it as the
    master document using the GNP file asset API.

    ``doc`` may be an already-open document owned by the calling thread; it is
    reused (and left open) instead of reopening the file per page.
    """
    temp_pdf_path = None
    owned_doc = doc is None
    new_doc = None
    try:
        if owned_doc:
            with _fitz_guard():
                doc = fitz.open(file_path)

        with _fitz_guard():
            page = doc.load_page(page_number - 1)

            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)

            temp_pdf_path = os.path.join(
                tempfile.gettempdir(),
                f"{asset_id}_page_{page_number}.pdf"
            )
            new_doc.save(temp_pdf_path)
            new_doc.close()
            new_doc = None
            if owned_doc:
                doc.close()
                doc = None

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
        # Only clean up what this call owns: the temp slice we produced and,
        # if we opened it, the document handle.
        if new_doc is not None:
            try:
                with _fitz_guard():
                    new_doc.close()
            except Exception:
                pass
        if owned_doc and doc is not None:
            try:
                with _fitz_guard():
                    doc.close()
            except Exception:
                pass
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass


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
        response = _post_json_with_retry(url, data, timeout=300)
        if response is None:
            logger.error("Failed to create enhanced publication: no response")
            return None

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


# ============================================================================
# SHARED PIPELINE INSTANCE
# ============================================================================
# A single EnhancedOCRPipeline can safely be shared by all page workers:
#   * AdvancedLayoutAnalyzer is stateless,
#   * ContentClassifier keeps a locked memo cache and one Ollama client per
#     thread.
# Sharing it also lets identical/repeated pages skip the LLM round-trip.

_shared_pipeline: Optional[EnhancedOCRPipeline] = None
_shared_pipeline_lock = threading.Lock()


def get_shared_pipeline(use_docling: bool = False) -> EnhancedOCRPipeline:
    """Return the process-wide OCR pipeline instance (thread-safe)."""
    global _shared_pipeline
    if _shared_pipeline is None:
        with _shared_pipeline_lock:
            if _shared_pipeline is None:
                _shared_pipeline = EnhancedOCRPipeline(use_docling=use_docling)
    return _shared_pipeline


# ============================================================================
# HTTP RETRY HELPER
# ============================================================================
# Running many requests in parallel multiplies the chance of hitting a
# transient 429/5xx, so the publication call is retried with jittered backoff.

_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _post_json_with_retry(url: str, payload: dict, timeout: int = 300,
                          attempts: int = None, base_delay: float = None):
    """POST JSON with bounded exponential backoff. Returns Response | None."""
    if attempts is None:
        attempts = int(os.getenv("OCR_HTTP_RETRIES", "3"))
    if base_delay is None:
        base_delay = float(os.getenv("OCR_HTTP_RETRY_DELAY", "2.0"))

    last_exc = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            response = None
        else:
            if response.status_code not in _RETRY_STATUS:
                return response

        if attempt >= attempts:
            break

        delay = base_delay * (2 ** (attempt - 1))
        delay += delay * 0.25 * (0.5 - os.urandom(1)[0] / 255.0)
        status = getattr(response, "status_code", "error")
        logger.warning(
            f"POST {url} failed with {status} "
            f"(attempt {attempt}/{attempts}); retrying in {delay:.1f}s"
        )
        time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    return response


# ============================================================================
# CONCURRENT PROCESSING ENGINE
# ============================================================================

class PdfDocumentPool:
    """
    Hand out one open ``fitz.Document`` per worker thread for a single file.

    MuPDF documents must not be shared between threads, so each worker opens
    its own handle.  All handles are closed together once every page of the
    file has been processed.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._docs: List["fitz.Document"] = []

    def get(self) -> "fitz.Document":
        doc = getattr(self._tls, "doc", None)
        if doc is None:
            with _fitz_guard():
                doc = fitz.open(self.file_path)
            self._tls.doc = doc
            with self._lock:
                self._docs.append(doc)
        return doc

    def close_all(self) -> None:
        with self._lock:
            docs, self._docs = self._docs, []
        for doc in docs:
            try:
                with _fitz_guard():
                    doc.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Error closing PDF handle: {exc}")


@dataclass
class FileContext:
    """Metadata shared by every worker handling a single source file."""

    file_path: str
    file_id: str
    file_type: str                       # "pdf" or "image"
    total_pages: int = 1
    edition_number: str = ""
    publication_date: str = ""
    doc_pool: Optional[PdfDocumentPool] = None

    @property
    def label(self) -> str:
        return os.path.basename(self.file_path)


class _FileTask:
    """
    Owns the in-flight token for one file, and (when it came through
    ``ConcurrentPipeline.submit_file``) the backpressure slot as well.

    Exactly one ``_FileTask`` is created per file, so ``finish()`` is called
    exactly once and ``wait()`` can never block on a stale token.
    """

    __slots__ = ("engine", "holds_slot", "_lock", "_finished")

    def __init__(self, engine: "ConcurrentPipeline", holds_slot: bool):
        self.engine = engine
        self.holds_slot = holds_slot
        self._lock = threading.Lock()
        self._finished = False
        engine._begin_file_task()

    def finish(self, ok: bool = True) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        self.engine._finish_file_task(ok=ok)
        if self.holds_slot:
            self.engine._release_slot()


class _FileJob:
    """
    Reference-counted handle for one file.

    The dispatch reference counts as 1; every queued page adds one more.
    Whoever drops the count to zero finalises the file: close PDF handles,
    delete the temporary download and finish the associated ``_FileTask``.
    """

    def __init__(self, engine: "ConcurrentPipeline", ctx: FileContext,
                 task: Optional[_FileTask] = None):
        self.engine = engine
        self.ctx = ctx
        self.task = task or _FileTask(engine, holds_slot=False)
        self._lock = threading.Lock()
        self._remaining = 1
        self._closed = False

    def add(self, n: int = 1) -> None:
        with self._lock:
            self._remaining += n

    def done(self) -> None:
        with self._lock:
            self._remaining -= 1
            should_close = self._remaining <= 0 and not self._closed
            if should_close:
                self._closed = True
        if should_close:
            self.engine._finalize_file(self.ctx, self.task, ok=True)


class ConcurrentPipeline:
    """Thread-pool driver: downloads files and OCRs their pages concurrently."""

    def __init__(self, page_workers: int = DEFAULT_PAGE_WORKERS,
                 file_workers: int = DEFAULT_FILE_WORKERS,
                 max_inflight_files: int = None,
                 keep_files: bool = False,
                 dry_run: bool = False):
        self.page_workers = max(1, int(page_workers))
        self.file_workers = max(1, int(file_workers))
        self.max_inflight_files = max(
            self.file_workers,
            int(max_inflight_files if max_inflight_files is not None
                else max(self.file_workers * 2, 4)),
        )
        self.keep_files = keep_files
        self.dry_run = dry_run

        self.page_executor = ThreadPoolExecutor(
            max_workers=self.page_workers, thread_name_prefix="ocr-page"
        )
        self.file_executor = ThreadPoolExecutor(
            max_workers=self.file_workers, thread_name_prefix="ocr-file"
        )

        self.counter = ProgressCounter(label="pages")
        self.files_done = 0
        self.files_failed = 0

        self._slots = threading.BoundedSemaphore(self.max_inflight_files)
        self._cond = threading.Condition()
        self._inflight_files = 0
        self._submitted_files = 0
        self._limit_reached = threading.Event()
        self._file_limit = None
        self._started = time.time()

    # -- scheduling ------------------------------------------------------
    def set_file_limit(self, limit: Optional[int]) -> None:
        self._file_limit = limit

    def submit_file(self, file_id: str, file_name: str) -> None:
        """Queue one downloaded file for processing."""
        if self._file_limit is not None:
            with self._cond:
                if self._submitted_files >= self._file_limit:
                    self._limit_reached.set()
                    return
                self._submitted_files += 1

        if self.dry_run:
            logger.info(f"[dry-run] would process {file_name}")
            task = _FileTask(self, holds_slot=False)
            task.finish(ok=True)
            return

        # Backpressure: block the traversal thread once too many files are
        # in flight, so we never queue an unbounded number of pages.
        self._slots.acquire()
        try:
            task = _FileTask(self, holds_slot=True)
            self.file_executor.submit(
                self._download_and_dispatch, file_id, file_name, task
            )
        except RuntimeError:
            # Executor already shut down.
            task.finish(ok=False)
            raise

    def _download_and_dispatch(self, file_id: str, file_name: str,
                               task: _FileTask) -> None:
        ctx: Optional[FileContext] = None
        job: Optional[_FileJob] = None
        try:
            path = download_file(None, file_id, file_name, process=False)
            if not path:
                return

            extension = os.path.splitext(path)[1].lower()
            if extension == ".pdf":
                ctx = self._prepare_pdf(path, file_id)
                if ctx is None:
                    return
                job = self._dispatch_pdf(ctx, task)
            elif extension in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"):
                ctx = FileContext(file_path=path, file_id=file_id, file_type="image")
                job = self._dispatch_pages(ctx, 1, task)
            else:
                logger.info(f"Skipping unsupported file type: {path}")
                ctx = FileContext(path, file_id, "unknown")
        except Exception as exc:
            logger.error(f"Error dispatching {file_name}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            # NOTE: when page work was queued it now owns ``task`` and will
            # finalise the file; releasing the dispatch reference here would
            # close documents and delete the download while the last page is
            # still running.
            if job is None:
                if ctx is not None:
                    self._finalize_file(ctx, task, ok=False)
                else:
                    task.finish(ok=False)

    def _prepare_pdf(self, path: str, file_id: str) -> Optional[FileContext]:
        """Read edition/date from page 1 before fanning out the other pages."""
        ctx = FileContext(file_path=path, file_id=file_id, file_type="pdf")
        try:
            with _fitz_guard():
                doc = fitz.open(path)
            try:
                ctx.total_pages = len(doc)
                with _fitz_guard():
                    page = doc.load_page(0)
                pipeline = get_shared_pipeline()
                text = pipeline.layout_analyzer.extract_text_with_layout(
                    path, 1, doc=doc
                )
                if not text or len(text.strip()) < 10:
                    with _fitz_guard():
                        text = page.get_text("text")
                ctx.edition_number = extract_edition_number(text) or ""
                ctx.publication_date = extract_publication_date(text) or ""
            finally:
                with _fitz_guard():
                    doc.close()
        except Exception as exc:
            logger.error(f"Failed to inspect PDF {path}: {exc}")
            return None

        logger.info(
            f"{os.path.basename(path)}: {ctx.total_pages} pages "
            f"(edition={ctx.edition_number or 'n/a'}, "
            f"date={ctx.publication_date or 'n/a'})"
        )
        return ctx

    def _dispatch_pdf(self, ctx: FileContext,
                      task: Optional[_FileTask] = None) -> _FileJob:
        ctx.doc_pool = PdfDocumentPool(ctx.file_path)
        return self._dispatch_pages(ctx, ctx.total_pages, task)

    def _dispatch_pages(self, ctx: FileContext, total: int,
                        task: Optional[_FileTask] = None) -> _FileJob:
        job = _FileJob(self, ctx, task)
        job.add(total)
        self.counter.add_total(total)

        for page_index in range(total):
            future = self.page_executor.submit(self._run_page, ctx, page_index, job)
            future.add_done_callback(_consume_future)

        # Give the dispatch reference back immediately: from here on the job
        # is kept alive purely by its page futures.
        job.done()
        return job

    # -- workers ---------------------------------------------------------
    def _run_page(self, ctx: FileContext, page_index: int, job: _FileJob) -> None:
        try:
            if ctx.file_type == "pdf":
                ok = process_pdf_page(ctx, page_index)
            else:
                ok = process_image_file(ctx)
            self.counter.tick(ok=ok)
        except Exception as exc:  # never let a page kill the pool
            logger.error(
                f"{ctx.label}: unhandled error on page {page_index + 1}: {exc}"
            )
            import traceback
            traceback.print_exc()
            self.counter.tick(ok=False)
        finally:
            job.done()

    # -- lifecycle -------------------------------------------------------
    def _begin_file_task(self) -> None:
        with self._cond:
            self._inflight_files += 1

    def _release_slot(self) -> None:
        try:
            self._slots.release()
        except ValueError:  # pragma: no cover - defensive
            pass

    def _finalize_file(self, ctx: FileContext, task: _FileTask,
                       ok: bool = True) -> None:
        if ctx.doc_pool is not None:
            ctx.doc_pool.close_all()

        if not self.keep_files:
            self._cleanup_file(ctx)
        elif ctx.file_path and os.path.exists(ctx.file_path):
            logger.info(f"Kept downloaded file: {ctx.file_path}")

        task.finish(ok=ok)

    def _cleanup_file(self, ctx: FileContext) -> None:
        if ctx.file_path and os.path.exists(ctx.file_path):
            try:
                os.remove(ctx.file_path)
                logger.debug(f"Cleaned up temporary file: {ctx.file_path}")
            except OSError as exc:
                logger.warning(f"Could not remove {ctx.file_path}: {exc}")

    def _finish_file_task(self, ok: bool = True) -> None:
        with self._cond:
            if ok:
                self.files_done += 1
            else:
                self.files_failed += 1
            self._inflight_files -= 1
            self._cond.notify_all()

    # -- waiting ---------------------------------------------------------
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until every submitted file (and its pages) has finished."""
        deadline = None if timeout is None else time.time() + timeout
        with self._cond:
            while self._inflight_files > 0:
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def shutdown(self) -> None:
        self.page_executor.shutdown(wait=True)
        self.file_executor.shutdown(wait=True)

    def summary(self) -> str:
        elapsed = time.time() - self._started
        rate = self.counter.done / elapsed if elapsed > 0 else 0.0
        return (
            f"Processed {self.files_done} file(s) / "
            f"{self.counter.done + self.counter.failed} page(s) in "
            f"{elapsed:.1f}s ({rate:.2f} pages/s, "
            f"{self.counter.failed} page failure(s), "
            f"{self.files_failed} file failure(s), "
            f"page_workers={self.page_workers}, "
            f"file_workers={self.file_workers})"
        )


def _consume_future(future: Future) -> None:
    """Surface exceptions raised by queued work instead of swallowing them."""
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:  # pragma: no cover - _run_page already catches
        logger.error(f"Page worker raised: {exc}")


_engine: Optional[ConcurrentPipeline] = None
_engine_lock = threading.Lock()


def get_engine(**kwargs) -> ConcurrentPipeline:
    """Return the process-wide pipeline engine, creating it on first use."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ConcurrentPipeline(**kwargs)
    return _engine


def init_engine(page_workers: int = DEFAULT_PAGE_WORKERS,
                file_workers: int = DEFAULT_FILE_WORKERS,
                **kwargs) -> ConcurrentPipeline:
    """(Re)create the shared engine with explicit settings."""
    global _engine
    with _engine_lock:
        _engine = ConcurrentPipeline(
            page_workers=page_workers, file_workers=file_workers, **kwargs
        )
    return _engine


# ============================================================================
# PER-PAGE WORKERS
# ============================================================================

def process_pdf_page(ctx: FileContext, page_index: int) -> bool:
    """OCR, upload and publish a single PDF page. Returns True on success."""
    page_number = page_index + 1
    label = f"{ctx.label} p{page_number}/{ctx.total_pages}"

    pipeline = get_shared_pipeline()
    doc = ctx.doc_pool.get()
    with _fitz_guard():
        page = doc.load_page(page_index)

    page_text = pipeline.layout_analyzer.extract_text_with_layout(
        ctx.file_path, page_number, doc=doc
    )
    if not page_text or len(page_text.strip()) < 10:
        with _fitz_guard():
            page_text = page.get_text("text")

    logger.info(f"{label}: extracted {len(page_text)} characters")

    classification_result = pipeline.process_ocr_text(page_text, ctx.label)

    # One UUID identifies the logical asset/publication. The same ID is used
    # for the thumbnail upload, the master document upload and the publication.
    asset_id = str(uuid.uuid4())

    # 1. Create and upload the obfuscated thumbnail.
    thumbnail_url = None
    thumbnail_buffer = obfuscate_pdf_thumbnail_smart(page, output_size=(500, 333))
    if thumbnail_buffer:
        thumbnail_url = upload_thumbnail_to_cloudinary(
            thumbnail_buffer, asset_id, page_num=page_number
        )
        if thumbnail_url:
            logger.info(f"{label}: thumbnail uploaded")
        else:
            logger.error(f"{label}: thumbnail upload failed")
    else:
        logger.error(f"{label}: failed to create thumbnail")

    # 2. Create and upload the single-page master document.
    master_result = upload_pdf_page(
        ctx.file_path, asset_id, page_number, doc=doc
    )
    if not master_result:
        logger.error(f"{label}: master upload failed; skipping publication")
        return False

    pub_date = ctx.publication_date or datetime.now().strftime("%Y-%m-%d")

    # 3. Create the publication entry.
    create_enhanced_publication(
        pub_date,
        str(page_number),
        classification_result,
        asset_id,
        "pdf",
        ctx.edition_number,
        thumbnail_url=thumbnail_url,
    )
    logger.info(f"{label}: publication created (asset_id={asset_id})")
    return True


def process_image_file(ctx: FileContext) -> bool:
    """OCR, upload and publish a single image. Returns True on success."""
    label = ctx.label
    pipeline = get_shared_pipeline()

    with Image.open(ctx.file_path) as img:
        extracted_text = pytesseract.image_to_string(img)

    classification_result = pipeline.process_ocr_text(extracted_text, ctx.file_path)

    date, page = extract_info(ctx.file_path)
    if not (date and page):
        logger.warning(
            f"{label}: could not extract date/page from filename; skipping"
        )
        return False

    asset_id = str(uuid.uuid4())

    # 1. Upload the obfuscated thumbnail.
    thumbnail_url = None
    thumbnail_buffer = create_obfuscated_thumbnail_from_image(
        ctx.file_path, method="balanced"
    )
    if thumbnail_buffer:
        thumbnail_url = upload_thumbnail_to_cloudinary(
            thumbnail_buffer, asset_id, page_num=1
        )
    if not thumbnail_url:
        logger.warning(f"{label}: thumbnail upload failed")

    # 2. Upload the original image as the master document.  Uploading from a
    #    handle (rather than a path) keeps ownership of the file with this
    #    worker, so the cleanup in _finalize_file stays authoritative.
    master_bucket = os.getenv(
        "GNP_MASTER_DOCUMENT_BUCKET", "gnp-master-documents"
    )
    lower_name = ctx.file_path.lower()
    content_type = (
        "image/jpeg" if lower_name.endswith((".jpg", ".jpeg")) else "image/png"
    )
    with open(ctx.file_path, "rb") as master_handle:
        master_result = upload_file_asset(
            file_content=master_handle,
            asset_id=asset_id,
            bucket_name=master_bucket,
            file_name=os.path.basename(ctx.file_path),
            content_type=content_type,
        )

    if not master_result:
        logger.error(f"{label}: master upload failed; skipping publication")
        return False

    # 3. Create the publication entry.
    create_enhanced_publication(
        date,
        page,
        classification_result,
        asset_id,
        "image",
        "N/A",
        thumbnail_url=thumbnail_url,
    )
    logger.info(f"{label}: publication created (asset_id={asset_id})")
    return True


# ============================================================================
# DRIVE DOWNLOAD
# ============================================================================

def download_file(service, file_id, file_name, process=True):
    """
    Download a file from Google Drive to ``file_name``.

    When ``process`` is True (the historical behaviour) the file is processed
    inline.  The concurrent engine calls it with ``process=False`` and
    dispatches the work itself.
    """
    service = service or get_drive_service()

    parent = os.path.dirname(file_name)
    if parent:
        os.makedirs(parent, exist_ok=True)

    attempts = int(os.getenv("OCR_DOWNLOAD_RETRIES", "3"))
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            with open(file_name, 'wb') as f:
                fh.seek(0)
                shutil.copyfileobj(fh, f)
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                logger.error(f"Download of {file_name} failed: {exc}")
                return None
            delay = min(30, 2 ** attempt)
            logger.warning(
                f"Download of {file_name} failed ({exc}); "
                f"retrying in {delay}s ({attempt}/{attempts})"
            )
            time.sleep(delay)

    logger.info(f"Downloaded {file_name}")

    if not process:
        return file_name

    file_path = os.path.join('.', file_name)
    _, extension = os.path.splitext(file_path)
    if extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        ocr_image(file_path, file_id, service)
    elif extension.lower() == '.pdf':
        process_pdf(file_path, file_id, service)
    return file_path


# ============================================================================
# BACKWARDS-COMPATIBLE ENTRY POINTS
# ============================================================================

def ocr_image(file_path, file_id, service=None):
    """Process a single image with OCR, upload assets, and create publication."""
    ctx = FileContext(file_path=file_path, file_id=file_id, file_type="image")
    engine = get_engine()
    try:
        ok = process_image_file(ctx)
        engine.counter.tick(ok=ok)
    except Exception as exc:
        engine.counter.tick(ok=False)
        logger.error(f"Error processing image: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if not engine.keep_files:
            engine._cleanup_file(ctx)


def process_pdf(file_path, file_id, service=None):
    """
    Process a PDF: upload obfuscated thumbnails and per-page master documents.

    Pages are processed on the engine's thread pool; this call blocks until
    every page of this file is done.
    """
    engine = get_engine()
    ctx = engine._prepare_pdf(file_path, file_id)
    if ctx is None:
        engine._cleanup_file(FileContext(file_path, file_id, "pdf"))
        return

    ctx.doc_pool = PdfDocumentPool(file_path)
    done = threading.Event()

    class _InlineJob:
        """Minimal stand-in for _FileJob so process_pdf can be called standalone."""

        def done(self):
            done.set()

    job = _InlineJob()
    for page_index in range(ctx.total_pages):
        future = engine.page_executor.submit(engine._run_page, ctx, page_index, job)
        future.add_done_callback(_consume_future)

    done.wait()
    ctx.doc_pool.close_all()
    if not engine.keep_files:
        engine._cleanup_file(ctx)


# ============================================================================
# DRIVE TRAVERSAL
# ============================================================================

def _list_children(service, folder_id: str):
    """Yield every child of a Drive folder, following pagination."""
    page_token = None
    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()

        for item in results.get('files', []):
            yield item

        page_token = results.get('nextPageToken')
        if not page_token:
            break


def traverse_and_download(service, folder_id, base_path=''):
    """
    Traverse Google Drive folders and queue every file for processing.

    Files are downloaded and OCR'd concurrently; this function returns as soon
    as everything has been queued (call ``engine.wait()`` to block).
    """
    engine = get_engine()
    for item in _list_children(service, folder_id):
        logger.info(f"Found {item['name']}")

        if item['mimeType'] == 'application/vnd.google-apps.folder':
            new_path = os.path.join(base_path, item['name'])
            os.makedirs(new_path, exist_ok=True)
            traverse_and_download(service, item['id'], new_path)
        else:
            file_path = os.path.join(base_path, item['name'])
            engine.submit_file(item['id'], file_path)

            if engine._limit_reached.is_set():
                logger.info("File limit reached; stopping traversal")
                return


def read_folder_ids(file_path):
    """Read folder IDs from file"""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]


# ============================================================================
# CONCURRENCY SETTINGS RESOLUTION
# ============================================================================
# Precedence, highest first:
#   1. --sequential      forces 1 page + 1 file worker (control run)
#   2. --cpus N          N single-threaded page workers
#   3. --page-workers N  explicit page worker count
#   4. defaults          CPU-bounded, see DEFAULT_* above
#
# ``--cpus`` and ``--page-workers`` are two spellings of the same knob: the
# worker count.  ``--omp-thread-limit`` is the *other* half of the CPU budget,
# so ``--cpus N`` simply pins it to 1.

@dataclass
class ConcurrencySettings:
    """Resolved CPU/concurrency budget for one run."""

    page_workers: int
    file_workers: int
    omp_thread_limit: int
    source: str                       # how page_workers was decided

    @property
    def cpu_budget(self) -> int:
        return self.page_workers * self.omp_thread_limit

    def describe(self) -> str:
        """One-line summary written to the log at startup."""
        detail = (
            f"{self.page_workers} page worker(s) x "
            f"{self.omp_thread_limit} tesseract thread(s) = "
            f"{self.cpu_budget} of {CPU_COUNT} CPU(s) "
            f"[{self.source}]; {self.file_workers} file worker(s)"
        )
        return detail

    def warnings(self) -> List[str]:
        """Human-readable problems with this budget, if any."""
        out = []
        if self.cpu_budget > CPU_COUNT:
            out.append(
                f"CPU budget {self.cpu_budget} exceeds the {CPU_COUNT} "
                f"available CPU(s): OCR workers will contend. Use --cpus "
                f"{CPU_COUNT}, or set --omp-thread-limit 1 and lower "
                f"--page-workers."
            )
        elif self.cpu_budget < CPU_COUNT * 0.5:
            out.append(
                f"CPU budget {self.cpu_budget} uses less than half of the "
                f"{CPU_COUNT} available CPU(s); you can likely raise "
                f"--page-workers to {CPU_COUNT}."
            )
        if self.page_workers > CPU_COUNT:
            out.append(
                f"--page-workers {self.page_workers} exceeds the {CPU_COUNT} "
                f"CPU(s); the per-page work is CPU-bound, so extra workers "
                f"mostly add context switching."
            )
        return out


def resolve_concurrency(page_workers: Optional[int] = None,
                        file_workers: Optional[int] = None,
                        sequential: bool = False,
                        cpus: Optional[int] = None,
                        omp_thread_limit: Optional[int] = None) -> ConcurrencySettings:
    """
    Turn user flags into a concrete concurrency budget.

    An explicit ``OMP_THREAD_LIMIT`` in the environment always wins over
    ``--omp-thread-limit`` so shell-level tuning stays authoritative.
    """
    env_limit = os.getenv("OMP_THREAD_LIMIT")

    if sequential:
        workers, source = 1, "--sequential"
    elif cpus is not None:
        workers, source = max(1, int(cpus)), "--cpus"
    elif page_workers is not None:
        workers, source = max(1, int(page_workers)), "--page-workers"
    else:
        workers, source = DEFAULT_PAGE_WORKERS, "default"

    if env_limit not in (None, ""):
        limit = max(1, int(env_limit))
    elif omp_thread_limit is not None:
        limit = max(1, int(omp_thread_limit))
    else:
        limit = 1

    files = 1 if sequential else max(1, int(file_workers or DEFAULT_FILE_WORKERS))

    return ConcurrencySettings(
        page_workers=workers,
        file_workers=files,
        omp_thread_limit=limit,
        source=source,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Archive OCR pipeline (concurrent Drive traversal + OCR)"
    )
    parser.add_argument(
        "-p", "--page-workers", type=int, default=DEFAULT_PAGE_WORKERS,
        help="Pages/images processed concurrently (default: %(default)s)",
    )
    parser.add_argument(
        "-f", "--file-workers", type=int, default=DEFAULT_FILE_WORKERS,
        help="Files downloaded and dispatched concurrently (default: %(default)s)",
    )
    parser.add_argument(
        "--max-inflight-files", type=int, default=None,
        help="Cap on files held in memory/on disk at once (backpressure)",
    )
    parser.add_argument(
        "--folder-ids", default="folder_ids.txt",
        help="Path to the file listing Drive folder IDs (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N files (useful for smoke tests)",
    )
    parser.add_argument(
        "--keep-files", action="store_true",
        help="Do not delete downloaded files after processing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be processed without downloading or uploading",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Disable concurrency (1 page + 1 file worker) as a control run",
    )
    parser.add_argument(
        "--omp-thread-limit", type=int, default=None,
        help=(
            "Threads per tesseract subprocess (default: 1). "
            "Roughly, page_workers x this should equal your CPU count; "
            "an OMP_THREAD_LIMIT env var takes precedence."
        ),
    )
    parser.add_argument(
        "--cpus", type=int, default=None,
        help=(
            "CPU budget to target: sets --page-workers to this many workers "
            "with OMP_THREAD_LIMIT=1 (overrides --page-workers)."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point"""
    args = parse_args(argv)

    settings = resolve_concurrency(
        page_workers=args.page_workers,
        file_workers=args.file_workers,
        sequential=args.sequential,
        cpus=args.cpus,
        omp_thread_limit=args.omp_thread_limit,
    )
    page_workers = settings.page_workers
    file_workers = settings.file_workers

    configure_tesseract_env(
        page_workers=page_workers,
        omp_thread_limit=settings.omp_thread_limit,
    )
    logger.info(f"Concurrency: {settings.describe()}")
    for problem in settings.warnings():
        logger.warning(problem)

    init_cloudinary()

    creds = auth_drive()
    set_drive_credentials(creds)
    service = get_drive_service()

    engine = init_engine(
        page_workers=page_workers,
        file_workers=file_workers,
        max_inflight_files=args.max_inflight_files,
        keep_files=args.keep_files,
        dry_run=args.dry_run,
    )
    engine.set_file_limit(args.limit)

    folder_ids = read_folder_ids(args.folder_ids)
    logger.info(
        f"Processing {len(folder_ids)} folder(s) with "
        f"{page_workers} page worker(s) and {file_workers} file worker(s)"
    )

    exit_code = 0
    try:
        for folder_id in folder_ids:
            logger.info(f"Traversing folder: {folder_id}")
            traverse_and_download(service, folder_id)
        engine.wait()
    except KeyboardInterrupt:
        logger.warning("Interrupted; waiting for in-flight work to finish")
        exit_code = 130
        engine.wait()
    finally:
        engine.shutdown()
        logger.info(engine.summary())
        if engine.counter.failed:
            exit_code = exit_code or 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
