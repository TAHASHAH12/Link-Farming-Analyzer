"""
LLM Model Probing & Link Authority Analysis Dashboard
PRODUCTION VERSION - Advanced Visualizations & Key Findings
Version: 4.0
Date: November 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
import time
from collections import Counter, defaultdict
import numpy as np
import hashlib
from io import StringIO
import base64
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse, urlunparse
import random
import chardet
from scipy import stats

# BeautifulSoup for parsing
from bs4 import BeautifulSoup

# Google Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="LLM Model Probing & Link Authority Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-left: 5px solid #1f77b4;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
    }
    .api-connected {
        background-color: #38ef7d;
        color: #1a5928;
    }
    .api-disconnected {
        background-color: #f5576c;
        color: white;
    }
    .finding-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .critical-finding {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'probe_results' not in st.session_state:
    st.session_state.probe_results = []
if 'link_analysis_results' not in st.session_state:
    st.session_state.link_analysis_results = []
if 'domain_summary' not in st.session_state:
    st.session_state.domain_summary = {}
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
if 'similarity_model' not in st.session_state:
    st.session_state.similarity_model = None
if 'similarity_model_loaded' not in st.session_state:
    st.session_state.similarity_model_loaded = False

# ==================== WEB SCRAPER (PREVIOUS CODE) ====================

class SequentialWebScraper:
    """Sequential web scraper with rate limiting"""
    
    def __init__(self, delay_between_requests: float = 2.0):
        self.session = self._create_session()
        self.delay = delay_between_requests
        self.cache = {}
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _get_random_headers(self) -> Dict:
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
    
    def _detect_encoding(self, content: bytes) -> str:
        try:
            result = chardet.detect(content)
            encoding = result['encoding']
            if encoding:
                return encoding
        except:
            pass
        return 'utf-8'
    
    def _decode_content(self, content: bytes) -> str:
        try:
            encoding = self._detect_encoding(content)
            return content.decode(encoding, errors='ignore')
        except:
            pass
        
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'ascii']:
            try:
                return content.decode(encoding, errors='ignore')
            except:
                continue
        
        return content.decode('utf-8', errors='ignore')
    
    def _clean_text(self, soup: BeautifulSoup) -> str:
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                            'iframe', 'noscript', 'svg', 'form', 'button', 'meta', 'link']):
            element.decompose()
        
        text = None
        
        for tag in ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.entry', '.article-body']:
            try:
                element = soup.select_one(tag)
                if element:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > 200:
                        break
            except:
                continue
        
        if not text or len(text) < 200:
            try:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
            except:
                pass
        
        if not text or len(text) < 100:
            try:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)
            except:
                pass
        
        if text:
            try:
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                text = text.encode('ascii', errors='ignore').decode('ascii')
                return text[:5000]
            except:
                return text[:5000]
        
        return ""
    
    def extract_text_basic(self, url: str, timeout: int = 15) -> Tuple[Optional[str], str]:
        if url in self.cache:
            logger.info(f"Cache hit for {url}")
            return self.cache[url]
        
        try:
            headers = self._get_random_headers()
            time.sleep(self.delay + random.uniform(0, 1))
            
            response = self.session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                result = (None, f"non_html")
                self.cache[url] = result
                return result
            
            html_content = self._decode_content(response.content)
            
            try:
                soup = BeautifulSoup(html_content, 'lxml')
            except:
                soup = BeautifulSoup(html_content, 'html.parser')
            
            text = self._clean_text(soup)
            
            if len(text) > 50:
                logger.info(f"✓ Extracted {len(text)} chars from {url}")
                result = (text, "basic_success")
                self.cache[url] = result
                return result
            
            logger.warning(f"Insufficient text from {url}")
            result = (None, "insufficient_text")
            self.cache[url] = result
            return result
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"http_{e.response.status_code}"
            logger.error(f"HTTP {e.response.status_code} for {url}")
            result = (None, error_msg)
            self.cache[url] = result
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for {url}")
            return (None, "timeout")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {url}")
            return (None, "connection_error")
        except Exception as e:
            logger.error(f"Error for {url}: {str(e)[:100]}")
            return (None, "error")
    
    def extract_text_selenium(self, url: str, timeout: int = 20) -> Tuple[Optional[str], str]:
        if not SELENIUM_AVAILABLE:
            return (None, "selenium_unavailable")
        
        driver = None
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument(f'user-agent={random.choice(self.user_agents)}')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--log-level=3')
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(timeout)
            
            time.sleep(self.delay + random.uniform(0, 1))
            
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
            
            page_source = driver.page_source
            html_content = self._decode_content(page_source.encode('utf-8'))
            
            try:
                soup = BeautifulSoup(html_content, 'lxml')
            except:
                soup = BeautifulSoup(html_content, 'html.parser')
            
            text = self._clean_text(soup)
            
            if len(text) > 50:
                logger.info(f"✓ Extracted {len(text)} chars from {url} (Selenium)")
                result = (text, "selenium_success")
                self.cache[url] = result
                return result
            
            return (None, "selenium_insufficient")
            
        except Exception as e:
            logger.error(f"Selenium error for {url}: {str(e)[:100]}")
            return (None, "selenium_error")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def extract_text_with_fallback(self, url: str) -> Tuple[Optional[str], str]:
        text, method = self.extract_text_basic(url)
        if text and len(text) > 50:
            return text, method
        
        logger.warning(f"Basic failed for {url}, trying Selenium...")
        
        text, method = self.extract_text_selenium(url)
        if text and len(text) > 50:
            return text, method
        
        logger.error(f"All methods failed for {url}")
        return None, "all_failed"
    
    def extract_bulk_sequential(self, urls: List[str], progress_callback=None) -> Dict[str, Tuple[Optional[str], str]]:
        results = {}
        total = len(urls)
        unique_urls = list(dict.fromkeys(urls))
        logger.info(f"Processing {len(unique_urls)} unique URLs sequentially...")
        
        for idx, url in enumerate(unique_urls):
            try:
                text, method = self.extract_text_with_fallback(url)
                results[url] = (text, method)
                
                if progress_callback:
                    progress_callback(idx + 1, total)
                
                logger.info(f"Progress: {idx+1}/{len(unique_urls)} URLs processed")
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results[url] = (None, "exception")
                
                if progress_callback:
                    progress_callback(idx + 1, total)
        
        return results

# ==================== GEMINI & SIMILARITY CLIENTS (PREVIOUS CODE) ====================

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.is_connected = False
        
        if GEMINI_AVAILABLE and api_key:
            try:
                os.environ["GEMINI_API_KEY"] = api_key
                self.client = genai.Client(api_key=api_key)
                self.is_connected = True
                logger.info("Gemini connected")
            except Exception as e:
                logger.error(f"Gemini init failed: {e}")
    
    def generate_response(self, prompt: str, model: str = "gemini-2.0-flash-exp", 
                         temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        if not self.is_connected:
            return None
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=1024,
                    )
                )
                
                if response and response.text:
                    return response.text.strip()
                    
            except Exception as e:
                logger.warning(f"Gemini call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def check_needs_retrieval(self, question: str, answer: str) -> bool:
        if not self.is_connected:
            return False
        
        prompt = f"""You were asked: "{question}"
You answered: "{answer}"

Did you rely only on prior knowledge, or would you need retrieval?
Respond with only 'PRIOR_KNOWLEDGE' or 'NEEDS_RETRIEVAL'."""
        
        try:
            response = self.generate_response(prompt, temperature=0.0)
            return 'NEEDS_RETRIEVAL' in response.upper() if response else False
        except:
            return False

class OfflineSemanticSimilarity:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", delay: float = 2.0):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self.scraper = SequentialWebScraper(delay_between_requests=delay)
    
    def load_model(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            logger.info("Model loaded")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False
    
    def calculate_similarity(self, text1: str, text2: str) -> Optional[float]:
        if not self.is_loaded:
            return None
        
        try:
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = util.cos_sim(embedding1, embedding2)
            return float(similarity.item())
        except Exception as e:
            logger.error(f"Similarity error: {e}")
            return None
    
    @staticmethod
    def _get_interpretation(score: float) -> str:
        if score >= 0.70:
            return "On-Topic"
        elif score >= 0.56:
            return "Borderline"
        elif score >= 0.36:
            return "Weak"
        else:
            return "Off-Topic"
    
    def get_similarity(self, external_url: str, target_url: str) -> Optional[Dict]:
        if not self.is_loaded:
            return None
        
        logger.info(f"Analyzing: {external_url} vs {target_url}")
        
        external_text, ext_method = self.scraper.extract_text_with_fallback(external_url)
        target_text, tgt_method = self.scraper.extract_text_with_fallback(target_url)
        
        if not external_text or not target_text:
            return {
                'external_url': external_url,
                'target_url': target_url,
                'similarity': 0.0,
                'interpretation': 'Error',
                'api_status': 'error',
                'http_status': 400,
                'timestamp': datetime.now().isoformat(),
                'extraction_methods': f"ext:{ext_method}, tgt:{tgt_method}",
                'raw': json.dumps({'error': 'Extraction failed'})
            }
        
        similarity = self.calculate_similarity(external_text, target_text)
        
        if similarity is None:
            similarity = 0.0
        
        return {
            'external_url': external_url,
            'target_url': target_url,
            'similarity': round(similarity, 3),
            'interpretation': self._get_interpretation(similarity),
            'api_status': 'success',
            'http_status': 200,
            'timestamp': datetime.now().isoformat(),
            'extraction_methods': f"ext:{ext_method}, tgt:{tgt_method}",
            'raw': json.dumps({'score': similarity, 'model': self.model_name})
        }
    
    def get_similarity_bulk(self, pairs: List[Tuple[str, str]], progress_callback=None) -> List[Dict]:
        if not self.is_loaded:
            return []
        
        logger.info(f"Bulk analysis of {len(pairs)} pairs (sequential)")
        
        all_urls = list(set([url for pair in pairs for url in pair]))
        logger.info(f"Extracting from {len(all_urls)} unique URLs...")
        
        def extraction_progress(current, total):
            if progress_callback:
                progress_callback(int(current / total * 70), 100)
        
        url_texts = self.scraper.extract_bulk_sequential(all_urls, extraction_progress)
        
        results = []
        total_pairs = len(pairs)
        
        for idx, (ext_url, tgt_url) in enumerate(pairs):
            if progress_callback:
                progress_callback(70 + int((idx + 1) / total_pairs * 30), 100)
            
            ext_text, ext_method = url_texts.get(ext_url, (None, 'not_found'))
            tgt_text, tgt_method = url_texts.get(tgt_url, (None, 'not_found'))
            
            if not ext_text or not tgt_text:
                results.append({
                    'external_url': ext_url,
                    'target_url': tgt_url,
                    'similarity': 0.0,
                    'interpretation': 'Error',
                    'api_status': 'error',
                    'http_status': 400,
                    'timestamp': datetime.now().isoformat(),
                    'extraction_methods': f"ext:{ext_method}, tgt:{tgt_method}",
                    'raw': json.dumps({'error': 'Extraction failed'})
                })
                continue
            
            similarity = self.calculate_similarity(ext_text, tgt_text)
            
            if similarity is None:
                similarity = 0.0
            
            results.append({
                'external_url': ext_url,
                'target_url': tgt_url,
                'similarity': round(similarity, 3),
                'interpretation': self._get_interpretation(similarity),
                'api_status': 'success',
                'http_status': 200,
                'timestamp': datetime.now().isoformat(),
                'extraction_methods': f"ext:{ext_method}, tgt:{tgt_method}",
                'raw': json.dumps({'score': similarity, 'model': self.model_name})
            })
        
        logger.info(f"Bulk complete: {len(results)} results")
        return results

# ==================== ADVANCED ANALYSIS FUNCTIONS ====================

def analyze_key_findings(df: pd.DataFrame, domain_summary: pd.DataFrame) -> Dict:
    """
    Generate comprehensive key findings from analysis
    """
    findings = {
        'overall_health': {},
        'critical_issues': [],
        'recommendations': [],
        'statistics': {},
        'risk_assessment': {}
    }
    
    # Overall statistics
    total_links = len(df)
    avg_similarity = df['similarity'].mean()
    median_similarity = df['similarity'].median()
    std_similarity = df['similarity'].std()
    
    # Interpretation breakdown
    interp_counts = df['interpretation'].value_counts()
    off_topic_count = interp_counts.get('Off-Topic', 0)
    on_topic_count = interp_counts.get('On-Topic', 0)
    
    # Domain analysis
    total_domains = len(domain_summary)
    flagged_domains = domain_summary['flagged'].sum()
    
    # Overall Health Score (0-100)
    health_score = (on_topic_count / total_links * 40) + \
                   ((1 - off_topic_count / total_links) * 30) + \
                   ((1 - flagged_domains / total_domains) * 30 if total_domains > 0 else 0)
    
    findings['overall_health'] = {
        'score': round(health_score, 1),
        'grade': 'A' if health_score >= 90 else 'B' if health_score >= 80 else 'C' if health_score >= 70 else 'D' if health_score >= 60 else 'F',
        'status': 'Excellent' if health_score >= 90 else 'Good' if health_score >= 80 else 'Fair' if health_score >= 70 else 'Poor' if health_score >= 60 else 'Critical'
    }
    
    # Statistics
    findings['statistics'] = {
        'total_links': total_links,
        'avg_similarity': round(avg_similarity, 3),
        'median_similarity': round(median_similarity, 3),
        'std_similarity': round(std_similarity, 3),
        'on_topic': on_topic_count,
        'off_topic': off_topic_count,
        'on_topic_rate': round(on_topic_count / total_links * 100, 1),
        'off_topic_rate': round(off_topic_count / total_links * 100, 1),
        'total_domains': total_domains,
        'flagged_domains': flagged_domains,
        'domain_flag_rate': round(flagged_domains / total_domains * 100, 1) if total_domains > 0 else 0
    }
    
    # Critical Issues
    if off_topic_count / total_links > 0.3:
        findings['critical_issues'].append({
            'severity': 'HIGH',
            'issue': 'High Off-Topic Rate',
            'detail': f"{round(off_topic_count / total_links * 100, 1)}% of links are off-topic (threshold: 30%)",
            'impact': 'Significant SEO penalty risk'
        })
    
    if flagged_domains > 0:
        findings['critical_issues'].append({
            'severity': 'HIGH',
            'issue': 'Suspect Link Farms Detected',
            'detail': f"{flagged_domains} domains flagged as potential link farms",
            'impact': 'Risk of manual action from search engines'
        })
    
    if avg_similarity < 0.5:
        findings['critical_issues'].append({
            'severity': 'MEDIUM',
            'issue': 'Low Average Similarity',
            'detail': f"Average similarity of {round(avg_similarity, 3)} is below recommended threshold of 0.5",
            'impact': 'Weak topical relevance of backlink profile'
        })
    
    # Recommendations
    if off_topic_count > 0:
        findings['recommendations'].append({
            'priority': 'HIGH',
            'action': 'Remove Off-Topic Links',
            'detail': f"Disavow or remove {off_topic_count} off-topic backlinks to improve link profile quality"
        })
    
    if flagged_domains > 0:
        findings['recommendations'].append({
            'priority': 'CRITICAL',
            'action': 'Address Link Farm Issues',
            'detail': f"Immediately review and disavow links from {flagged_domains} flagged domains"
        })
    
    if on_topic_count / total_links > 0.7:
        findings['recommendations'].append({
            'priority': 'LOW',
            'action': 'Maintain Quality Standards',
            'detail': "Continue building high-quality, topically relevant backlinks"
        })
    else:
        findings['recommendations'].append({
            'priority': 'MEDIUM',
            'action': 'Improve Link Relevance',
            'detail': "Focus on acquiring links from contextually relevant sources"
        })
    
    # Risk Assessment
    risk_level = 'LOW'
    if health_score < 60 or flagged_domains > 3:
        risk_level = 'CRITICAL'
    elif health_score < 70 or flagged_domains > 0:
        risk_level = 'HIGH'
    elif health_score < 80:
        risk_level = 'MEDIUM'
    
    findings['risk_assessment'] = {
        'level': risk_level,
        'manual_action_risk': 'High' if flagged_domains > 2 else 'Medium' if flagged_domains > 0 else 'Low',
        'penalty_risk': 'High' if off_topic_count / total_links > 0.4 else 'Medium' if off_topic_count / total_links > 0.2 else 'Low'
    }
    
    return findings

def generate_advanced_report(df: pd.DataFrame, domain_summary: pd.DataFrame, findings: Dict) -> str:
    """
    Generate comprehensive markdown report with findings
    """
    report = f"# Advanced Link Authority Analysis Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Total Links Analyzed:** {len(df)}\n"
    report += f"**Total Domains:** {len(domain_summary)}\n\n"
    report += "---\n\n"
    
    # Overall Health
    report += "## 📊 Overall Health Score\n\n"
    health = findings['overall_health']
    report += f"**Score:** {health['score']}/100 (Grade: {health['grade']})\n"
    report += f"**Status:** {health['status']}\n\n"
    
    # Statistics
    report += "## 📈 Key Statistics\n\n"
    stats = findings['statistics']
    report += f"- **Total Links:** {stats['total_links']}\n"
    report += f"- **Average Similarity:** {stats['avg_similarity']}\n"
    report += f"- **Median Similarity:** {stats['median_similarity']}\n"
    report += f"- **Standard Deviation:** {stats['std_similarity']}\n"
    report += f"- **On-Topic Links:** {stats['on_topic']} ({stats['on_topic_rate']}%)\n"
    report += f"- **Off-Topic Links:** {stats['off_topic']} ({stats['off_topic_rate']}%)\n"
    report += f"- **Flagged Domains:** {stats['flagged_domains']}/{stats['total_domains']} ({stats['domain_flag_rate']}%)\n\n"
    
    # Critical Issues
    if findings['critical_issues']:
        report += "## 🚨 Critical Issues\n\n"
        for issue in findings['critical_issues']:
            report += f"### {issue['severity']}: {issue['issue']}\n\n"
            report += f"**Detail:** {issue['detail']}\n"
            report += f"**Impact:** {issue['impact']}\n\n"
    
    # Recommendations
    report += "## 💡 Recommendations\n\n"
    for rec in findings['recommendations']:
        report += f"### [{rec['priority']}] {rec['action']}\n\n"
        report += f"{rec['detail']}\n\n"
    
    # Risk Assessment
    report += "## ⚠️ Risk Assessment\n\n"
    risk = findings['risk_assessment']
    report += f"**Overall Risk Level:** {risk['level']}\n"
    report += f"**Manual Action Risk:** {risk['manual_action_risk']}\n"
    report += f"**Penalty Risk:** {risk['penalty_risk']}\n\n"
    
    return report

# ==================== ADVANCED VISUALIZATIONS ====================

def create_advanced_similarity_analysis(df: pd.DataFrame):
    """
    Create comprehensive similarity analysis visualization
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution', 'Box Plot', 'Violin Plot', 'Cumulative Distribution'),
        specs=[[{"type": "histogram"}, {"type": "box"}],
               [{"type": "violin"}, {"type": "scatter"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df['similarity'], nbinsx=30, name='Distribution',
                    marker_color='#667eea', showlegend=False),
        row=1, col=1
    )
    
    # Box Plot
    fig.add_trace(
        go.Box(y=df['similarity'], name='Box Plot',
              marker_color='#764ba2', showlegend=False),
        row=1, col=2
    )
    
    # Violin Plot
    fig.add_trace(
        go.Violin(y=df['similarity'], name='Violin Plot',
                 fillcolor='#38ef7d', line_color='#11998e', showlegend=False),
        row=2, col=1
    )
    
    # Cumulative Distribution
    sorted_sim = np.sort(df['similarity'])
    cumulative = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim) * 100
    fig.add_trace(
        go.Scatter(x=sorted_sim, y=cumulative, mode='lines',
                  name='Cumulative %', line=dict(color='#f5576c', width=3),
                  showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Advanced Similarity Score Analysis",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Similarity Score", row=1, col=1)
    fig.update_xaxes(title_text="Similarity Score", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Similarity Score", row=1, col=2)
    fig.update_yaxes(title_text="Similarity Score", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=2, col=2)
    
    return fig

def create_interpretation_sunburst(df: pd.DataFrame):
    """
    Create sunburst chart for interpretation breakdown
    """
    # Add score ranges
    df_copy = df.copy()
    df_copy['score_range'] = pd.cut(
        df_copy['similarity'],
        bins=[0, 0.35, 0.56, 0.70, 1.0],
        labels=['0.00-0.35', '0.36-0.55', '0.56-0.69', '0.70-1.00']
    )
    
    # Create hierarchy data
    data = df_copy.groupby(['interpretation', 'score_range']).size().reset_index(name='count')
    
    fig = px.sunburst(
        data,
        path=['interpretation', 'score_range'],
        values='count',
        color='interpretation',
        color_discrete_map={
            'On-Topic': '#38ef7d',
            'Borderline': '#ffd93d',
            'Weak': '#ff9a76',
            'Off-Topic': '#f5576c'
        },
        title="Link Quality Hierarchy"
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_domain_treemap(domain_summary: pd.DataFrame):
    """
    Create treemap visualization for domain analysis
    """
    top_domains = domain_summary.head(20).copy()
    top_domains['label'] = top_domains['external_domain'] + '<br>' + top_domains['avg_score'].apply(lambda x: f"{x:.3f}")
    
    fig = px.treemap(
        top_domains,
        path=[px.Constant("All Domains"), 'verdict', 'external_domain'],
        values='pair_count',
        color='avg_score',
        color_continuous_scale='RdYlGn',
        title="Domain Authority Treemap (Top 20)",
        hover_data=['pair_count', 'off_topic_rate']
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame):
    """
    Create correlation heatmap if there's enough data
    """
    # Extract numerical features
    df_copy = df.copy()
    df_copy['interp_numeric'] = df_copy['interpretation'].map({
        'On-Topic': 4,
        'Borderline': 3,
        'Weak': 2,
        'Off-Topic': 1
    })
    
    # Simple correlation matrix
    corr_data = pd.DataFrame({
        'Similarity': df_copy['similarity'],
        'Quality': df_copy['interp_numeric']
    })
    
    corr_matrix = corr_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlGn',
        text=corr_matrix.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=400
    )
    
    return fig

def create_score_timeline(df: pd.DataFrame):
    """
    Create timeline of scores (if timestamp available)
    """
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy = df_copy.sort_values('timestamp')
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=df_copy['timestamp'],
        y=df_copy['similarity'],
        mode='markers',
        name='Similarity Scores',
        marker=dict(
            size=8,
            color=df_copy['similarity'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Similarity")
        ),
        text=df_copy['interpretation'],
        hovertemplate='<b>Time:</b> %{x}<br><b>Similarity:</b> %{y:.3f}<br><b>Type:</b> %{text}<extra></extra>'
    ))
    
    # Add moving average
    window = min(10, len(df_copy) // 10)
    if window > 1:
        df_copy['ma'] = df_copy['similarity'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=df_copy['timestamp'],
            y=df_copy['ma'],
            mode='lines',
            name=f'{window}-Period Moving Average',
            line=dict(color='#667eea', width=3)
        ))
    
    fig.update_layout(
        title="Similarity Score Timeline",
        xaxis_title="Time",
        yaxis_title="Similarity Score",
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_statistical_summary_table(df: pd.DataFrame):
    """
    Create detailed statistical summary table
    """
    summary_stats = df['similarity'].describe()
    
    # Additional statistics
    skewness = df['similarity'].skew()
    kurtosis = df['similarity'].kurtosis()
    q1 = df['similarity'].quantile(0.25)
    q3 = df['similarity'].quantile(0.75)
    iqr = q3 - q1
    
    summary_data = {
        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max', 'IQR', 'Skewness', 'Kurtosis'],
        'Value': [
            f"{int(summary_stats['count'])}",
            f"{summary_stats['mean']:.4f}",
            f"{summary_stats['std']:.4f}",
            f"{summary_stats['min']:.4f}",
            f"{q1:.4f}",
            f"{summary_stats['50%']:.4f}",
            f"{q3:.4f}",
            f"{summary_stats['max']:.4f}",
            f"{iqr:.4f}",
            f"{skewness:.4f}",
            f"{kurtosis:.4f}"
        ]
    }
    
    return pd.DataFrame(summary_data)


# ==================== HELPER FUNCTIONS (REST OF PREVIOUS CODE) ====================

def get_similarity_score_bucket(score: float) -> Tuple[str, str]:
    if score >= 0.70:
        return "On-Topic", "score-high"
    elif score >= 0.56:
        return "Borderline", "score-medium"
    elif score >= 0.36:
        return "Weak", "score-medium"
    else:
        return "Off-Topic", "score-low"

@st.cache_resource
def load_similarity_model(model_name: str = "all-MiniLM-L6-v2", delay: float = 2.0):
    client = OfflineSemanticSimilarity(model_name, delay)
    client.load_model()
    return client

def run_probe_with_paraphrases(gemini_client, question, paraphrases, url, model, temperature, k_runs):
    all_questions = [question] + paraphrases
    results = {
        'url': url,
        'original_question': question,
        'paraphrases': paraphrases,
        'responses': defaultdict(list),
        'consistency_score': 0,
        'paraphrase_robust': False,
        'needs_retrieval': False,
        'model_used': model
    }
    
    prefix = """Answer using ONLY existing training knowledge. Do NOT simulate browsing.
If you don't know, say "I don't have this information."

Question: """
    
    for q in all_questions:
        for run in range(k_runs):
            response = gemini_client.generate_response(prefix + q, model, temperature)
            results['responses'][q].append(response if response else "API_ERROR")
            time.sleep(0.5)
    
    all_responses = [r for responses in results['responses'].values() for r in responses]
    valid_responses = [r for r in all_responses if r != "API_ERROR"]
    
    if valid_responses:
        most_common = Counter(valid_responses).most_common(1)[0]
        results['consistency_score'] = (most_common[1] / len(valid_responses)) * 100
        results['most_common_answer'] = most_common[0]
        results['paraphrase_robust'] = results['consistency_score'] >= 80
        results['needs_retrieval'] = gemini_client.check_needs_retrieval(question, most_common[0])
    else:
        results['most_common_answer'] = "All API calls failed"
        results['needs_retrieval'] = True
    
    return results

def analyze_domain_pattern(df, min_pairs=3, off_topic_threshold=0.6, avg_score_threshold=0.45):
    df['external_domain'] = df['external_url'].apply(
        lambda x: urlparse(x).netloc if '://' in x else x.split('/')[0]
    )
    
    domain_groups = df.groupby('external_domain').agg({
        'similarity': ['mean', 'min', 'max', 'count'],
        'interpretation': lambda x: (x == 'Off-Topic').sum()
    }).reset_index()
    
    domain_groups.columns = ['external_domain', 'avg_score', 'min_score', 
                              'max_score', 'pair_count', 'off_topic_count']
    
    domain_groups['off_topic_rate'] = domain_groups['off_topic_count'] / domain_groups['pair_count']
    domain_groups['flagged'] = (
        (domain_groups['pair_count'] >= min_pairs) &
        (domain_groups['off_topic_rate'] >= off_topic_threshold) &
        (domain_groups['avg_score'] <= avg_score_threshold)
    )
    domain_groups['verdict'] = domain_groups.apply(
        lambda row: 'SUSPECT LINK FARM' if row['flagged'] else 'OK', axis=1
    )
    
    return domain_groups.sort_values('avg_score')

def generate_markdown_report(probe_results, url):
    report = f"# LLM Model Probing Report\n\n**Target URL:** {url}\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    
    for idx, r in enumerate(probe_results, 1):
        report += f"## Probe {idx}: {r['original_question']}\n\n"
        report += f"**Consistency:** {r['consistency_score']:.1f}%\n"
        report += f"**Answer:** {r.get('most_common_answer', 'N/A')}\n\n---\n\n"
    
    return report

def generate_cmseo_findings(domain_summary):
    flagged = domain_summary[domain_summary['flagged'] == True]
    
    report = f"# CMSEO Link Authority Findings\n\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Domains:** {len(domain_summary)}\n"
    report += f"**Flagged:** {len(flagged)}\n\n---\n\n"
    
    if len(flagged) > 0:
        report += "## Suspect Link Farms\n\n"
        for _, row in flagged.iterrows():
            report += f"### {row['external_domain']}\n"
            report += f"- Avg: {row['avg_score']:.3f}\n"
            report += f"- Off-Topic: {row['off_topic_rate']:.1%}\n\n"
    
    return report

def create_consistency_chart(probe_results):
    if not probe_results:
        return go.Figure()
    
    questions = [r['original_question'][:50] + "..." for r in probe_results]
    scores = [r['consistency_score'] for r in probe_results]
    colors = ['#38ef7d' if s >= 80 else '#ffd93d' if s >= 60 else '#f5576c' for s in scores]
    
    fig = go.Figure(data=[go.Bar(
        x=scores, y=questions, orientation='h',
        marker=dict(color=colors),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Response Consistency",
        xaxis_title="Consistency (%)",
        height=max(400, len(probe_results) * 80),
        showlegend=False
    )
    
    return fig

def create_similarity_distribution(df):
    fig = go.Figure(data=[go.Histogram(
        x=df['similarity'], nbinsx=20,
        marker=dict(color='#667eea')
    )])
    
    fig.add_vline(x=0.35, line_dash="dash", line_color="red")
    fig.add_vline(x=0.70, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title="Similarity Distribution",
        xaxis_title="Similarity",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_interpretation_pie(df):
    counts = df['interpretation'].value_counts()
    colors = {'On-Topic': '#38ef7d', 'Borderline': '#ffd93d', 
              'Weak': '#ff9a76', 'Off-Topic': '#f5576c'}
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=[colors.get(l, '#ccc') for l in counts.index]),
        hole=0.4
    )])
    
    fig.update_layout(title="Link Quality Distribution", height=400)
    return fig

def create_domain_comparison(domain_summary):
    top = domain_summary.head(15)
    colors = ['#f5576c' if f else '#38ef7d' for f in top['flagged']]
    
    fig = go.Figure(data=[go.Bar(
        x=top['avg_score'], y=top['external_domain'],
        orientation='h', marker=dict(color=colors),
        text=[f"{s:.3f}" for s in top['avg_score']]
    )])
    
    fig.update_layout(
        title="Domain Authority Scores",
        xaxis_title="Avg Similarity",
        height=max(400, len(top) * 40)
    )
    
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    st.markdown('<h1 class="main-header">🔍 LLM Model Probing & Link Authority Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🚀 Production Dashboard with Advanced Analytics:</strong>
        <ul>
            <li><strong>Model Probing:</strong> Test Gemini's knowledge with closed-book evaluation</li>
            <li><strong>Link Authority:</strong> Semantic similarity with rate limiting (Sequential processing)</li>
            <li><strong>Advanced Visualizations:</strong> Sunburst, treemap, correlation, timeline charts</li>
            <li><strong>Key Findings:</strong> Automated risk assessment and recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar (same as before with minor adjustments)
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        st.markdown("#### 🤖 Google Gemini")
        gemini_key = st.text_input("Gemini API Key", type="password", 
                                   value=st.session_state.gemini_api_key, key="sidebar_gemini_key")
        if gemini_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_key
        
        gemini_client = GeminiClient(st.session_state.gemini_api_key)
        
        if gemini_client.is_connected:
            st.markdown('<div class="api-status api-connected">✓ Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status api-disconnected">✗ Disconnected</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("#### 🤖 Semantic Similarity")
        model_options = {
            "all-MiniLM-L6-v2 (Fast)": "all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (Accurate)": "all-mpnet-base-v2",
        }
        selected = st.selectbox("Model", list(model_options.keys()), key="sidebar_model_select")
        model_name = model_options[selected]
        
        delay = st.slider("Delay Between Requests (seconds)", 1.0, 10.0, 2.0, 0.5, key="sidebar_delay",
                         help="Increase to avoid IP blocking")
        
        if st.button("🔄 Load Model", use_container_width=True, key="sidebar_load_model"):
            with st.spinner("Loading..."):
                sim_client = load_similarity_model(model_name, delay)
                st.session_state.similarity_model = sim_client
                st.session_state.similarity_model_loaded = sim_client.is_loaded
                if sim_client.is_loaded:
                    st.success("✅ Loaded!")
                else:
                    st.error("❌ Failed")
        
        if st.session_state.similarity_model_loaded:
            st.markdown('<div class="api-status api-connected">✓ Model Loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status api-disconnected">✗ Not Loaded</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        model_choice = st.selectbox("Gemini Model", 
            ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"], key="sidebar_gemini_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="sidebar_temperature")
        k_runs = st.number_input("Consistency Runs", 1, 10, 3, key="sidebar_k_runs")
        
        st.markdown("---")
        
        st.markdown("### 🔗 Link Analysis")
        min_pairs = st.number_input("Min Pairs", 1, 10, 3, key="sidebar_min_pairs")
        off_topic_threshold = st.slider("Off-Topic Threshold", 0.0, 1.0, 0.6, 0.05, key="sidebar_off_topic")
        avg_score_threshold = st.slider("Avg Score Threshold", 0.0, 1.0, 0.45, 0.05, key="sidebar_avg_score")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Data", use_container_width=True, key="sidebar_clear"):
            st.session_state.probe_results = []
            st.session_state.link_analysis_results = []
            st.session_state.domain_summary = {}
            st.rerun()
    
    # Main tabs - ENHANCED with new tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧪 Model Probing",
        "🔗 Link Authority",
        "📊 Advanced Visualizations",
        "🔍 Key Findings",
        "📈 Analytics Dashboard"
    ])
    
    # TAB 1: Model Probing (same as before)
    with tab1:
        st.markdown('<h2 class="sub-header">Model Probing</h2>', unsafe_allow_html=True)
        
        if not gemini_client.is_connected:
            st.error("⚠️ Gemini not connected")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_url = st.text_input("Target URL", key="probe_target_url")
        with col2:
            page_name = st.text_input("Page Name", key="probe_page_name")
        
        num_questions = st.number_input("Number of Facts", 1, 10, 3, key="probe_num_questions")
        
        questions_data = []
        for i in range(num_questions):
            with st.expander(f"📝 Fact #{i+1}", expanded=(i==0)):
                main_q = st.text_area(f"Question {i+1}", key=f"probe_q_{i}", height=80)
                expected = st.text_input("Expected Answer", key=f"probe_exp_{i}")
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    para1 = st.text_input("Paraphrase 1", key=f"probe_p1_{i}")
                with col_p2:
                    para2 = st.text_input("Paraphrase 2", key=f"probe_p2_{i}")
                
                if main_q:
                    questions_data.append({
                        'main': main_q,
                        'paraphrases': [p for p in [para1, para2] if p],
                        'expected': expected
                    })
        
        if st.button("🚀 Run Probe", type="primary", key="probe_run_button"):
            if gemini_client.is_connected and target_url and questions_data:
                with st.spinner("Running probes..."):
                    progress = st.progress(0)
                    results = []
                    
                    for idx, q_data in enumerate(questions_data):
                        result = run_probe_with_paraphrases(
                            gemini_client, q_data['main'], q_data['paraphrases'],
                            target_url, model_choice, temperature, k_runs
                        )
                        result['expected'] = q_data['expected']
                        result['page_name'] = page_name
                        results.append(result)
                        progress.progress((idx + 1) / len(questions_data))
                    
                    st.session_state.probe_results = results
                    st.success(f"✅ Completed {len(results)} probes!")
                    st.rerun()
            else:
                st.error("❌ Check configuration")
        
        if st.session_state.probe_results:
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            results = st.session_state.probe_results
            avg_cons = np.mean([r['consistency_score'] for r in results])
            robust = sum([r['paraphrase_robust'] for r in results])
            needs_ret = sum([r['needs_retrieval'] for r in results])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{len(results)}</h3><p>Probes</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{avg_cons:.1f}%</h3><p>Avg Consistency</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="success-card"><h3>{robust}/{len(results)}</h3><p>Robust</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="warning-card"><h3>{needs_ret}</h3><p>Need Retrieval</p></div>', unsafe_allow_html=True)
            
            fig = create_consistency_chart(results)
            st.plotly_chart(fig, use_container_width=True)
            
            report = generate_markdown_report(results, target_url)
            st.download_button("📄 Download Report", report, 
                             f"probe_report_{datetime.now().strftime('%Y%m%d')}.md",
                             key="probe_download_report")
    
    # TAB 2: Link Authority (same structure but enhanced display)
    with tab2:
        st.markdown('<h2 class="sub-header">Link Authority (Sequential Processing)</h2>', unsafe_allow_html=True)
        
        if not st.session_state.similarity_model_loaded:
            st.error("⚠️ Load model first")
        
        st.markdown(f"""
        <div class="info-box">
            <strong>🚀 Sequential Processing:</strong> Rate-limited extraction with {delay}s delays to avoid IP blocking.
            Includes encoding error handling for international characters.
        </div>
        """, unsafe_allow_html=True)
        
        mode = st.radio("Mode", ["Single Pair", "Bulk CSV"], horizontal=True, key="link_mode")
        
        if mode == "Single Pair":
            col1, col2 = st.columns(2)
            with col1:
                ext_url = st.text_input("External URL", key="link_ext_url")
            with col2:
                tgt_url = st.text_input("Target URL", key="link_tgt_url")
            
            if st.button("🚀 Analyze", key="link_analyze_button"):
                if st.session_state.similarity_model_loaded and ext_url and tgt_url:
                    with st.spinner("Analyzing..."):
                        sim_client = st.session_state.similarity_model
                        result = sim_client.get_similarity(ext_url, tgt_url)
                        
                        if result:
                            col_r1, col_r2, col_r3 = st.columns(3)
                            with col_r1:
                                st.metric("Similarity", f"{result['similarity']:.3f}")
                            with col_r2:
                                bucket, _ = get_similarity_score_bucket(result['similarity'])
                                st.markdown(f'<div class="info-card">{bucket}</div>', unsafe_allow_html=True)
                            with col_r3:
                                verdict = "⚠️ OFF" if result['similarity'] <= 0.45 else "✓ OK"
                                st.markdown(f'<div class="success-card">{verdict}</div>', unsafe_allow_html=True)
                            
                            st.json(result)
                        else:
                            st.error("❌ Failed")
                else:
                    st.error("❌ Check config")
        
        else:  # Bulk
            st.markdown("#### 📤 Bulk Upload")
            
            if st.button("📥 Template", key="link_download_template"):
                template = pd.DataFrame({
                    'external_url': ['https://example.com/page1'],
                    'target_url': ['https://target.com/page1']
                })
                csv = template.to_csv(index=False)
                st.download_button("Download template", csv, "pairs_template.csv", 
                                 key="link_download_template_file")
            
            uploaded = st.file_uploader("Upload pairs.csv", type=['csv'], key="link_file_uploader")
            
            if uploaded:
                try:
                    pairs_df = pd.read_csv(uploaded)
                    
                    if 'external_url' not in pairs_df.columns or 'target_url' not in pairs_df.columns:
                        st.error("❌ CSV must have 'external_url' and 'target_url' columns")
                    else:
                        st.success(f"✅ Loaded {len(pairs_df)} pairs")
                        st.dataframe(pairs_df.head(10))
                        
                        st.warning(f"⏱️ Estimated time: ~{len(pairs_df) * delay * 2 / 60:.1f} minutes (with {delay}s delays)")
                        
                        if st.button("🚀 Run Bulk (Sequential)", type="primary", key="link_bulk_analyze"):
                            if st.session_state.similarity_model_loaded:
                                sim_client = st.session_state.similarity_model
                                sim_client.scraper.delay = delay
                                
                                progress_bar = st.progress(0)
                                status = st.empty()
                                
                                def progress_callback(current, total):
                                    progress_bar.progress(current / 100)
                                    status.text(f"Progress: {current}%")
                                
                                with st.spinner("Processing sequentially..."):
                                    pairs = list(zip(pairs_df['external_url'], pairs_df['target_url']))
                                    results = sim_client.get_similarity_bulk(pairs, progress_callback)
                                    
                                    st.session_state.link_analysis_results = results
                                    status.text("")
                                    st.success(f"✅ Analyzed {len(results)} pairs!")
                                    st.rerun()
                            else:
                                st.error("❌ Load model first")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.session_state.link_analysis_results:
            st.markdown("---")
            st.markdown("### 📊 Basic Results")
            
            results_df = pd.DataFrame(st.session_state.link_analysis_results)
            
            avg_sim = results_df['similarity'].mean()
            off_topic = (results_df['interpretation'] == 'Off-Topic').sum()
            on_topic = (results_df['interpretation'] == 'On-Topic').sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{len(results_df)}</h3><p>Pairs</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{avg_sim:.3f}</h3><p>Avg</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="success-card"><h3>{on_topic}</h3><p>On-Topic</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="warning-card"><h3>{off_topic}</h3><p>Off-Topic</p></div>', unsafe_allow_html=True)
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig_dist = create_similarity_distribution(results_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            with col_v2:
                fig_pie = create_interpretation_pie(results_df)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.dataframe(results_df[['external_url', 'target_url', 'similarity', 
                                     'interpretation', 'extraction_methods']])
            
            csv = results_df.to_csv(index=False)
            st.download_button("📊 Download results.csv", csv, 
                             f"results_{datetime.now().strftime('%Y%m%d')}.csv",
                             key="link_download_results")
    
    # TAB 3: ADVANCED VISUALIZATIONS (NEW)
    with tab3:
        st.markdown('<h2 class="sub-header">Advanced Visualizations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.link_analysis_results:
            st.info("Run Link Authority Analysis first to see advanced visualizations")
        else:
            results_df = pd.DataFrame(st.session_state.link_analysis_results)
            
            # Advanced Similarity Analysis
            st.markdown("### 📊 Comprehensive Similarity Analysis")
            fig_advanced = create_advanced_similarity_analysis(results_df)
            st.plotly_chart(fig_advanced, use_container_width=True)
            
            # Sunburst Chart
            st.markdown("### 🌅 Quality Hierarchy (Sunburst)")
            fig_sunburst = create_interpretation_sunburst(results_df)
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            # Timeline
            st.markdown("### 📈 Score Timeline")
            fig_timeline = create_score_timeline(results_df)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Statistical Summary Table
            st.markdown("### 📋 Statistical Summary")
            stats_table = create_statistical_summary_table(results_df)
            st.dataframe(stats_table, use_container_width=True, hide_index=True)
            
            # Domain Treemap (if domain analysis done)
            if isinstance(st.session_state.domain_summary, pd.DataFrame):
                st.markdown("### 🗺️ Domain Authority Treemap")
                fig_treemap = create_domain_treemap(st.session_state.domain_summary)
                st.plotly_chart(fig_treemap, use_container_width=True)
    
    # TAB 4: KEY FINDINGS (NEW)
    with tab4:
        st.markdown('<h2 class="sub-header">Key Findings & Recommendations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.link_analysis_results:
            st.info("Run Link Authority Analysis first to generate key findings")
        else:
            results_df = pd.DataFrame(st.session_state.link_analysis_results)
            
            # Generate domain summary if not already done
            if not isinstance(st.session_state.domain_summary, pd.DataFrame):
                with st.spinner("Analyzing domains..."):
                    domain_summary = analyze_domain_pattern(
                        results_df, min_pairs, off_topic_threshold, avg_score_threshold
                    )
                    st.session_state.domain_summary = domain_summary
            else:
                domain_summary = st.session_state.domain_summary
            
            # Generate key findings
            findings = analyze_key_findings(results_df, domain_summary)
            
            # Overall Health Score
            st.markdown("### 🏥 Overall Health Score")
            health = findings['overall_health']
            
            col_health1, col_health2, col_health3 = st.columns(3)
            with col_health1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="font-size:4rem; margin:0;">{health['score']}</h2>
                    <p style="font-size:1.5rem; margin:0;">/100</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_health2:
                st.markdown(f"""
                <div class="info-card">
                    <h3>Grade</h3>
                    <h2 style="font-size:3rem; margin:0;">{health['grade']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_health3:
                status_class = "success-card" if health['score'] >= 80 else "warning-card" if health['score'] >= 60 else "critical-finding"
                st.markdown(f"""
                <div class="{status_class}">
                    <h3>Status</h3>
                    <h2>{health['status']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Critical Issues
            if findings['critical_issues']:
                st.markdown("### 🚨 Critical Issues")
                for issue in findings['critical_issues']:
                    severity_color = "#f5576c" if issue['severity'] == 'HIGH' else "#ffd93d"
                    st.markdown(f"""
                    <div class="critical-finding" style="background: linear-gradient(135deg, {severity_color} 0%, {severity_color}dd 100%);">
                        <h4>[{issue['severity']}] {issue['issue']}</h4>
                        <p><strong>Detail:</strong> {issue['detail']}</p>
                        <p><strong>Impact:</strong> {issue['impact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    <h3>✅ No Critical Issues Found</h3>
                    <p>Your link profile is in good health!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            for rec in findings['recommendations']:
                priority_color = {"CRITICAL": "#f5576c", "HIGH": "#ff9a76", "MEDIUM": "#ffd93d", "LOW": "#38ef7d"}
                color = priority_color.get(rec['priority'], "#667eea")
                
                st.markdown(f"""
                <div class="finding-box" style="border-left-color: {color};">
                    <h4>[{rec['priority']}] {rec['action']}</h4>
                    <p>{rec['detail']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk Assessment
            st.markdown("### ⚠️ Risk Assessment")
            risk = findings['risk_assessment']
            
            risk_colors = {"CRITICAL": "#f5576c", "HIGH": "#ff9a76", "MEDIUM": "#ffd93d", "LOW": "#38ef7d"}
            risk_color = risk_colors.get(risk['level'], "#667eea")
            
            st.markdown(f"""
            <div class="critical-finding" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}dd 100%);">
                <h3>Overall Risk Level: {risk['level']}</h3>
                <p><strong>Manual Action Risk:</strong> {risk['manual_action_risk']}</p>
                <p><strong>Penalty Risk:</strong> {risk['penalty_risk']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Statistics
            st.markdown("### 📊 Key Statistics")
            stats = findings['statistics']
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Links", stats['total_links'])
                st.metric("Avg Similarity", f"{stats['avg_similarity']:.3f}")
            with col_s2:
                st.metric("On-Topic", f"{stats['on_topic']} ({stats['on_topic_rate']}%)")
                st.metric("Off-Topic", f"{stats['off_topic']} ({stats['off_topic_rate']}%)")
            with col_s3:
                st.metric("Total Domains", stats['total_domains'])
                st.metric("Flagged Domains", f"{stats['flagged_domains']} ({stats['domain_flag_rate']}%)")
            with col_s4:
                st.metric("Median Similarity", f"{stats['median_similarity']:.3f}")
                st.metric("Std Deviation", f"{stats['std_similarity']:.3f}")
            
            # Download Advanced Report
            st.markdown("### 💾 Export Advanced Report")
            advanced_report = generate_advanced_report(results_df, domain_summary, findings)
            st.download_button(
                "📄 Download Advanced Report (Markdown)",
                advanced_report,
                f"advanced_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                key="download_advanced_report"
            )
    
    # TAB 5: Analytics Dashboard (Enhanced from previous version)
    with tab5:
        st.markdown('<h2 class="sub-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        has_probe = len(st.session_state.probe_results) > 0
        has_link = len(st.session_state.link_analysis_results) > 0
        
        if not has_probe and not has_link:
            st.info("No data yet. Run analyses to see comprehensive dashboard.")
        else:
            st.markdown("### 📈 Overview")
            
            cols = st.columns(3)
            with cols[0]:
                if has_probe:
                    avg_cons = np.mean([r['consistency_score'] for r in st.session_state.probe_results])
                    st.markdown(f'<div class="info-card"><h4>Probing</h4><p>{len(st.session_state.probe_results)} probes</p><p>Avg: {avg_cons:.1f}%</p></div>', unsafe_allow_html=True)
            
            with cols[1]:
                if has_link:
                    results_df = pd.DataFrame(st.session_state.link_analysis_results)
                    st.markdown(f'<div class="info-card"><h4>Link Analysis</h4><p>{len(results_df)} pairs</p><p>Avg: {results_df["similarity"].mean():.3f}</p></div>', unsafe_allow_html=True)
            
            with cols[2]:
                if isinstance(st.session_state.domain_summary, pd.DataFrame):
                    domain_summary = st.session_state.domain_summary
                    flagged = domain_summary['flagged'].sum()
                    st.markdown(f'<div class="warning-card"><h4>Domains</h4><p>{len(domain_summary)} analyzed</p><p>{flagged} flagged</p></div>', unsafe_allow_html=True)
            
            # Domain Patterns Section
            if isinstance(st.session_state.domain_summary, pd.DataFrame):
                st.markdown("---")
                st.markdown("### 📊 Domain Patterns")
                
                domain_summary = st.session_state.domain_summary
                
                total = len(domain_summary)
                flagged = domain_summary['flagged'].sum()
                worst = domain_summary['avg_score'].min()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>{total}</h3><p>Domains</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="warning-card"><h3>{flagged}</h3><p>Flagged</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="info-card"><h3>{worst:.3f}</h3><p>Worst</p></div>', unsafe_allow_html=True)
                with col4:
                    flag_rate = (flagged / total * 100) if total > 0 else 0
                    st.markdown(f'<div class="metric-card"><h3>{flag_rate:.1f}%</h3><p>Flag Rate</p></div>', unsafe_allow_html=True)
                
                fig = create_domain_comparison(domain_summary)
                st.plotly_chart(fig, use_container_width=True)
                
                flagged_df = domain_summary[domain_summary['flagged'] == True]
                
                if len(flagged_df) > 0:
                    st.markdown("### 🚨 Suspect Domains")
                    for idx, row in flagged_df.iterrows():
                        with st.expander(f"⚠️ {row['external_domain']}", key=f"domain_exp_{idx}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg", f"{row['avg_score']:.3f}")
                            with col2:
                                st.metric("Off-Topic", f"{row['off_topic_rate']:.1%}")
                            with col3:
                                st.metric("Pairs", int(row['pair_count']))
                else:
                    st.markdown('<div class="success-card"><h3>✅ No Suspect Domains</h3></div>', unsafe_allow_html=True)
                
                st.dataframe(domain_summary)
                
                # Downloads
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv = domain_summary.to_csv(index=False)
                    st.download_button("📊 Download domain_summary.csv", csv, 
                                     f"domain_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                                     key="domain_download_summary")
                with col_dl2:
                    findings = generate_cmseo_findings(domain_summary)
                    st.download_button("📄 Download findings.md", findings,
                                     f"findings_{datetime.now().strftime('%Y%m%d')}.md",
                                     key="domain_download_findings")

if __name__ == "__main__":
    main()
