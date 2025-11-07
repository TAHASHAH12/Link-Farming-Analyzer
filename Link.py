"""
LLM Model Probing & Link Authority Analysis Dashboard - PRODUCTION VERSION
Complete Streamlit Application with Google Gemini API and Offline Semantic Similarity
Author: Advanced Analytics Team
Date: November 2025
Version: 2.1 (Production - Offline Similarity)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
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

# Google Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Google Gemini SDK not installed. Run: pip install google-genai")

# Sentence Transformers for offline semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    from bs4 import BeautifulSoup
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Sentence Transformers not installed. Run: pip install sentence-transformers beautifulsoup4")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTab {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .score-high {
        background-color: #38ef7d;
        color: #1a5928;
    }
    .score-medium {
        background-color: #ffd93d;
        color: #6b5b00;
    }
    .score-low {
        background-color: #f5576c;
        color: white;
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
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
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

# ==================== API CLIENT CLASSES ====================

class GeminiClient:
    """Google Gemini API Client for LLM Probing"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.is_connected = False
        
        if GEMINI_AVAILABLE and api_key:
            try:
                os.environ["GEMINI_API_KEY"] = api_key
                self.client = genai.Client(api_key=api_key)
                self.is_connected = True
                logger.info("Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.is_connected = False
    
    def generate_response(self, prompt: str, model: str = "gemini-2.0-flash-exp", 
                         temperature: float = 0.1, max_retries: int = 3) -> Optional[str]:
        """Generate response from Gemini API with retry logic"""
        if not self.is_connected or not self.client:
            logger.error("Gemini client not connected")
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
                    logger.info(f"Gemini response generated successfully (attempt {attempt + 1})")
                    return response.text.strip()
                    
            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    return None
        
        return None
    
    def check_needs_retrieval(self, question: str, answer: str) -> bool:
        """Ask Gemini if it needed retrieval to answer the question"""
        if not self.is_connected:
            return False
        
        self_check_prompt = f"""You were asked: "{question}"
You answered: "{answer}"

Did you rely only on your prior training knowledge to answer this, or would you need to retrieve/search for this specific information to be confident? 
Respond with only 'PRIOR_KNOWLEDGE' or 'NEEDS_RETRIEVAL'."""
        
        try:
            response = self.generate_response(self_check_prompt, temperature=0.0)
            if response and 'NEEDS_RETRIEVAL' in response.upper():
                return True
        except Exception as e:
            logger.warning(f"Self-check failed: {e}")
        
        return False


class OfflineSemanticSimilarity:
    """Offline Semantic Similarity using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize offline semantic similarity model
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Sentence Transformers not available")
            return False
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def extract_text_from_url(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Extract text content from a URL
        
        Args:
            url: URL to extract text from
            timeout: Request timeout in seconds
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit to first 5000 characters for efficiency
            text = text[:5000]
            
            logger.info(f"Extracted {len(text)} characters from {url}")
            return text
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout extracting text from {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting text from {url}: {e}")
            return None
    
    def calculate_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1) or None if failed
        """
        if not self.is_loaded or not self.model:
            logger.error("Model not loaded")
            return None
        
        try:
            # Generate embeddings
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(embedding1, embedding2)
            
            # Convert to float
            score = float(similarity.item())
            
            logger.info(f"Calculated similarity: {score:.3f}")
            return score
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return None
    
    def get_similarity(self, external_url: str, target_url: str) -> Optional[Dict]:
        """
        Get semantic similarity score between two URLs
        
        Args:
            external_url: The external/linking page URL
            target_url: The target/linked page URL
            
        Returns:
            Dictionary with similarity results or None if failed
        """
        if not self.is_loaded:
            logger.error("Model not loaded")
            return None
        
        logger.info(f"Analyzing similarity between {external_url} and {target_url}")
        
        # Extract text from both URLs
        external_text = self.extract_text_from_url(external_url)
        target_text = self.extract_text_from_url(target_url)
        
        if not external_text or not target_text:
            logger.error("Could not extract text from one or both URLs")
            return {
                'external_url': external_url,
                'target_url': target_url,
                'similarity': 0.0,
                'interpretation': 'Error',
                'api_status': 'error',
                'http_status': 400,
                'timestamp': datetime.now().isoformat(),
                'raw': json.dumps({'error': 'Failed to extract text'})
            }
        
        # Calculate similarity
        similarity_score = self.calculate_similarity(external_text, target_text)
        
        if similarity_score is None:
            return None
        
        interpretation = self._get_interpretation(similarity_score)
        
        return {
            'external_url': external_url,
            'target_url': target_url,
            'similarity': round(similarity_score, 3),
            'interpretation': interpretation,
            'api_status': 'success',
            'http_status': 200,
            'timestamp': datetime.now().isoformat(),
            'raw': json.dumps({
                'score': similarity_score,
                'model': self.model_name,
                'method': 'offline'
            })
        }
    
    @staticmethod
    def _get_interpretation(score: float) -> str:
        """Convert similarity score to interpretation bucket"""
        if score >= 0.70:
            return "On-Topic"
        elif score >= 0.56:
            return "Borderline"
        elif score >= 0.36:
            return "Weak"
        else:
            return "Off-Topic"

# ==================== HELPER FUNCTIONS ====================

def get_similarity_score_bucket(score: float) -> Tuple[str, str]:
    """Categorize similarity score into buckets"""
    if score >= 0.70:
        return "On-Topic", "score-high"
    elif score >= 0.56:
        return "Borderline", "score-medium"
    elif score >= 0.36:
        return "Weak", "score-medium"
    else:
        return "Off-Topic", "score-low"

@st.cache_resource
def load_similarity_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the similarity model"""
    similarity_client = OfflineSemanticSimilarity(model_name)
    similarity_client.load_model()
    return similarity_client

def run_probe_with_paraphrases(
    gemini_client: GeminiClient,
    question: str, 
    paraphrases: List[str], 
    url: str,
    model: str, 
    temperature: float, 
    k_runs: int
) -> Dict:
    """Run probe with multiple paraphrases and k runs for consistency using Gemini"""
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
    
    closed_book_prefix = """Answer this question using ONLY your existing training knowledge. 
Do NOT simulate browsing or retrieval. If you don't know from your training data, say "I don't have this information in my training data."

Question: """
    
    total_calls = len(all_questions) * k_runs
    call_count = 0
    
    for q in all_questions:
        full_prompt = closed_book_prefix + q
        
        for run in range(k_runs):
            call_count += 1
            logger.info(f"Probe call {call_count}/{total_calls}: {q[:50]}...")
            
            response = gemini_client.generate_response(
                full_prompt,
                model=model,
                temperature=temperature
            )
            
            if response:
                results['responses'][q].append(response)
            else:
                results['responses'][q].append("API_ERROR")
            
            time.sleep(0.5)
    
    # Calculate consistency
    all_responses = []
    for q_responses in results['responses'].values():
        all_responses.extend(q_responses)
    
    valid_responses = [r for r in all_responses if r != "API_ERROR"]
    
    if valid_responses:
        most_common = Counter(valid_responses).most_common(1)[0]
        results['consistency_score'] = (most_common[1] / len(valid_responses)) * 100
        results['most_common_answer'] = most_common[0]
        results['paraphrase_robust'] = results['consistency_score'] >= 80
        
        results['needs_retrieval'] = gemini_client.check_needs_retrieval(
            question,
            results['most_common_answer']
        )
    else:
        results['most_common_answer'] = "All API calls failed"
        results['needs_retrieval'] = True
    
    return results

def analyze_domain_pattern(
    df: pd.DataFrame, 
    min_pairs: int = 3, 
    off_topic_threshold: float = 0.6,
    avg_score_threshold: float = 0.45
) -> pd.DataFrame:
    """Analyze link patterns by external domain to flag suspect link farms"""
    df['external_domain'] = df['external_url'].apply(
        lambda x: x.split('/')[2] if '://' in x else x.split('/')[0]
    )
    
    domain_groups = df.groupby('external_domain').agg({
        'similarity': ['mean', 'min', 'max', 'count'],
        'interpretation': lambda x: (x == 'Off-Topic').sum()
    }).reset_index()
    
    domain_groups.columns = [
        'external_domain', 'avg_score', 'min_score', 
        'max_score', 'pair_count', 'off_topic_count'
    ]
    
    domain_groups['off_topic_rate'] = (
        domain_groups['off_topic_count'] / domain_groups['pair_count']
    )
    
    domain_groups['flagged'] = (
        (domain_groups['pair_count'] >= min_pairs) &
        (domain_groups['off_topic_rate'] >= off_topic_threshold) &
        (domain_groups['avg_score'] <= avg_score_threshold)
    )
    
    domain_groups['verdict'] = domain_groups.apply(
        lambda row: 'SUSPECT LINK FARM' if row['flagged'] else 'OK', 
        axis=1
    )
    
    return domain_groups.sort_values('avg_score')

def generate_markdown_report(probe_results: List[Dict], url: str) -> str:
    """Generate markdown report for probe results"""
    report = f"# LLM Model Probing Report\n\n"
    report += f"**Target URL:** {url}\n"
    report += f"**Model:** {probe_results[0].get('model_used', 'N/A') if probe_results else 'N/A'}\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "---\n\n"
    
    for idx, result in enumerate(probe_results, 1):
        report += f"## Probe {idx}: {result['original_question']}\n\n"
        report += f"**Consistency Score:** {result['consistency_score']:.1f}%\n"
        report += f"**Most Common Answer:** {result.get('most_common_answer', 'N/A')}\n"
        report += f"**Paraphrase Robust:** {'✓ Yes' if result['paraphrase_robust'] else '✗ No'}\n"
        report += f"**Needs Retrieval:** {'✓ Yes' if result['needs_retrieval'] else '✗ No'}\n\n"
        
        if result['paraphrases']:
            report += "**Paraphrases tested:**\n"
            for para in result['paraphrases']:
                report += f"- {para}\n"
        
        report += "\n### Response Distribution:\n"
        for q, responses in result['responses'].items():
            if q == result['original_question']:
                report += f"\n**Original Question Responses ({len(responses)} runs):**\n"
                for i, resp in enumerate(responses, 1):
                    report += f"{i}. {resp[:100]}...\n"
        
        report += "\n---\n\n"
    
    return report

def generate_cmseo_findings(domain_summary: pd.DataFrame) -> str:
    """Generate CMSEO findings report"""
    flagged = domain_summary[domain_summary['flagged'] == True]
    
    report = "# CMSEO Link Authority Findings\n\n"
    report += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Analysis Method:** Offline Semantic Similarity (Sentence Transformers)\n"
    report += f"**Total Domains Analyzed:** {len(domain_summary)}\n"
    report += f"**Flagged Domains:** {len(flagged)}\n\n"
    report += "---\n\n"
    
    report += "## Flagging Criteria\n\n"
    report += "- Minimum 3 pairs from same domain\n"
    report += "- Off-topic rate ≥ 60%\n"
    report += "- Average similarity score ≤ 0.45\n\n"
    report += "---\n\n"
    
    if len(flagged) > 0:
        report += "## Suspect Link Farms Identified\n\n"
        for _, row in flagged.iterrows():
            report += f"### {row['external_domain']}\n\n"
            report += f"- **Average Score:** {row['avg_score']:.3f}\n"
            report += f"- **Score Range:** {row['min_score']:.3f} - {row['max_score']:.3f}\n"
            report += f"- **Off-Topic Rate:** {row['off_topic_rate']:.1%}\n"
            report += f"- **Total Pairs:** {int(row['pair_count'])}\n"
            report += f"- **Off-Topic Count:** {int(row['off_topic_count'])}\n"
            report += f"- **Verdict:** 🚨 {row['verdict']}\n\n"
            report += "**Recommendation:** Consider removing or devaluing links from this domain.\n\n"
    else:
        report += "## ✓ No Suspect Domains Found\n\n"
        report += "All analyzed domains meet quality thresholds.\n\n"
    
    report += "---\n\n"
    report += "## Summary Statistics\n\n"
    report += f"- **Best Domain Score:** {domain_summary['avg_score'].max():.3f}\n"
    report += f"- **Worst Domain Score:** {domain_summary['avg_score'].min():.3f}\n"
    report += f"- **Average Domain Score:** {domain_summary['avg_score'].mean():.3f}\n"
    report += f"- **Median Domain Score:** {domain_summary['avg_score'].median():.3f}\n"
    
    return report

# ==================== VISUALIZATION FUNCTIONS ====================

def create_consistency_chart(probe_results: List[Dict]) -> go.Figure:
    """Create consistency score visualization"""
    if not probe_results:
        return go.Figure()
    
    questions = [
        r['original_question'][:50] + "..." if len(r['original_question']) > 50 
        else r['original_question'] 
        for r in probe_results
    ]
    consistency_scores = [r['consistency_score'] for r in probe_results]
    colors = [
        '#38ef7d' if s >= 80 else '#ffd93d' if s >= 60 else '#f5576c' 
        for s in consistency_scores
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=consistency_scores,
            y=questions,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=2)
            ),
            text=[f"{s:.1f}%" for s in consistency_scores],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Consistency: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Response Consistency Across Runs",
        xaxis_title="Consistency Score (%)",
        yaxis_title="Question",
        height=max(400, len(probe_results) * 80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(range=[0, 105]),
        showlegend=False
    )
    
    return fig

def create_similarity_distribution(df: pd.DataFrame) -> go.Figure:
    """Create similarity score distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['similarity'],
        nbinsx=20,
        marker=dict(
            color='#667eea',
            line=dict(color='white', width=1)
        ),
        name='Similarity Scores',
        hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.add_vline(
        x=0.35, line_dash="dash", line_color="red", line_width=2,
        annotation_text="Off-Topic Threshold (0.35)",
        annotation_position="top"
    )
    fig.add_vline(
        x=0.70, line_dash="dash", line_color="green", line_width=2,
        annotation_text="On-Topic Threshold (0.70)",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_domain_comparison(domain_summary: pd.DataFrame) -> go.Figure:
    """Create domain comparison chart"""
    top_domains = domain_summary.head(15)
    
    colors = [
        '#f5576c' if flagged else '#38ef7d' 
        for flagged in top_domains['flagged']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_domains['avg_score'],
        y=top_domains['external_domain'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{s:.3f}" for s in top_domains['avg_score']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Avg Score: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Domain Authority Scores (Bottom 15)",
        xaxis_title="Average Similarity Score",
        yaxis_title="External Domain",
        height=max(400, len(top_domains) * 40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_interpretation_pie(df: pd.DataFrame) -> go.Figure:
    """Create interpretation bucket pie chart"""
    bucket_counts = df['interpretation'].value_counts()
    
    colors = {
        'On-Topic': '#38ef7d',
        'Borderline': '#ffd93d',
        'Weak': '#ff9a76',
        'Off-Topic': '#f5576c'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=bucket_counts.index,
        values=bucket_counts.values,
        marker=dict(
            colors=[colors.get(label, '#cccccc') for label in bucket_counts.index]
        ),
        textinfo='label+percent+value',
        textfont=dict(size=14),
        hole=0.4,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Link Quality Distribution",
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_heatmap_analysis(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of similarity scores"""
    if len(df) == 0:
        return go.Figure()
    
    df['external_short'] = df['external_url'].apply(
        lambda x: x.split('/')[2] if '://' in x else x[:30]
    )
    df['target_short'] = df['target_url'].apply(
        lambda x: '/'.join(x.split('/')[-2:]) if '/' in x else x[:20]
    )
    
    pivot = df.pivot_table(
        values='similarity', 
        index='external_short', 
        columns='target_short', 
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=np.round(pivot.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Similarity"),
        hovertemplate='External: %{y}<br>Target: %{x}<br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Similarity Heatmap: External vs Target URLs",
        height=max(400, len(pivot) * 30),
        xaxis_title="Target URL",
        yaxis_title="External Domain"
    )
    
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🔍 LLM Model Probing & Link Authority Dashboard</h1>', 
        unsafe_allow_html=True
    )
    
    st.markdown("""
    <div class="info-box">
        <strong>📊 Production Dashboard (Offline Mode):</strong> 
        <ul>
            <li><strong>Model Probing:</strong> Test LLM knowledge with Google Gemini API (closed-book evaluation)</li>
            <li><strong>Link Authority:</strong> Analyze semantic similarity using <strong>offline Sentence Transformers</strong></li>
            <li><strong>Domain Analysis:</strong> Identify suspect link farms and off-authority patterns</li>
        </ul>
        <p><strong>✨ No external API needed for similarity analysis!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR CONFIGURATION ====================
    with st.sidebar:
        st.markdown("### ⚙️ API Configuration")
        
        # Gemini API Configuration
        st.markdown("#### 🤖 Google Gemini API")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your API key from https://aistudio.google.com/app/apikey"
        )
        
        if gemini_api_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_api_key
        
        gemini_client = GeminiClient(st.session_state.gemini_api_key)
        
        if gemini_client.is_connected:
            st.markdown(
                '<div class="api-status api-connected">✓ Gemini Connected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="api-status api-disconnected">✗ Gemini Disconnected</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Offline Semantic Similarity Configuration
        st.markdown("#### 🤖 Offline Semantic Similarity")
        
        similarity_model_options = {
            "all-MiniLM-L6-v2 (Fast, 80MB)": "all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (Accurate, 420MB)": "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2 (Balanced, 80MB)": "paraphrase-MiniLM-L6-v2",
        }
        
        selected_model_display = st.selectbox(
            "Embedding Model",
            options=list(similarity_model_options.keys()),
            help="Choose embedding model for semantic similarity"
        )
        selected_model = similarity_model_options[selected_model_display]
        
        if st.button("🔄 Load Similarity Model", use_container_width=True):
            with st.spinner("Loading model... This may take a minute on first run..."):
                similarity_client = load_similarity_model(selected_model)
                st.session_state.similarity_model = similarity_client
                st.session_state.similarity_model_loaded = similarity_client.is_loaded
                if similarity_client.is_loaded:
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("❌ Failed to load model")
        
        if st.session_state.similarity_model_loaded:
            st.markdown(
                f'<div class="api-status api-connected">✓ Model Loaded: {selected_model}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="api-status api-disconnected">✗ Model Not Loaded</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🤖 LLM Configuration")
        model_choice = st.selectbox(
            "Gemini Model",
            [
                "gemini-2.0-flash-exp",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash-8b"
            ],
            help="Select Gemini model for probing"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower temperature = more consistent responses"
        )
        
        k_runs = st.number_input(
            "Consistency Runs (k)",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of times to run each probe"
        )
        
        st.markdown("---")
        
        # Link Analysis Configuration
        st.markdown("### 🔗 Link Analysis Config")
        min_pairs = st.number_input(
            "Min Pairs for Pattern",
            min_value=1,
            max_value=10,
            value=3,
            help="Minimum pairs needed to flag a domain"
        )
        
        off_topic_threshold = st.slider(
            "Off-Topic Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum off-topic rate to flag"
        )
        
        avg_score_threshold = st.slider(
            "Avg Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Maximum avg score to flag"
        )
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.probe_results = []
            st.session_state.link_analysis_results = []
            st.session_state.domain_summary = {}
            st.rerun()
    
    # ==================== MAIN TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Model Probing",
        "🔗 Link Authority Analysis (Offline)",
        "📊 Domain Patterns",
        "📈 Analytics Dashboard"
    ])
    
    # ==================== TAB 1: MODEL PROBING ====================
    with tab1:
        st.markdown('<h2 class="sub-header">LLM Model Probing with Google Gemini</h2>', 
                    unsafe_allow_html=True)
        
        if not gemini_client.is_connected:
            st.error("⚠️ Gemini API not connected. Please configure API key in sidebar.")
        
        st.markdown("""
        <div class="info-box">
            <strong>🎯 Closed-Book Testing:</strong> Evaluate what Gemini already knows about specific 
            pages without allowing web browsing or retrieval. Test consistency across multiple runs 
            and paraphrases.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_url = st.text_input(
                "Target URL (e.g., Stake Blackjack page)",
                placeholder="https://stake.com/casino/games/blackjack",
                help="The page you want to probe the LLM's knowledge about"
            )
        
        with col2:
            page_name = st.text_input(
                "Page Name",
                placeholder="Main Blackjack",
                help="Short identifier for this page"
            )
        
        st.markdown("#### Define Atomic Facts to Probe")
        
        num_questions = st.number_input(
            "Number of Facts to Probe",
            min_value=1,
            max_value=10,
            value=3,
            help="How many atomic facts do you want to test?"
        )
        
        questions_data = []
        for i in range(num_questions):
            with st.expander(f"📝 Fact #{i+1}", expanded=(i==0)):
                col_q1, col_q2 = st.columns([3, 2])
                
                with col_q1:
                    main_q = st.text_area(
                        f"Main Question",
                        placeholder="From your existing knowledge only, what blackjack payout is stated on Stake's main blackjack page?",
                        key=f"main_q_{i}",
                        height=100
                    )
                
                with col_q2:
                    expected = st.text_input(
                        "Expected Answer (optional)",
                        placeholder="3:2",
                        key=f"expected_{i}"
                    )
                
                st.markdown("**Paraphrases (for robustness testing):**")
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    para1 = st.text_input(
                        "Paraphrase 1",
                        placeholder="State the blackjack payout ratio Stake highlights",
                        key=f"para1_{i}"
                    )
                
                with col_p2:
                    para2 = st.text_input(
                        "Paraphrase 2",
                        placeholder="What's the blackjack payout mentioned on Stake?",
                        key=f"para2_{i}"
                    )
                
                if main_q:
                    paraphrases = [p for p in [para1, para2] if p]
                    questions_data.append({
                        'main': main_q,
                        'paraphrases': paraphrases,
                        'expected': expected
                    })
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            run_probe = st.button("🚀 Run Probe", use_container_width=True, type="primary")
        
        with col_btn2:
            if st.session_state.probe_results:
                clear_probe = st.button("Clear Results", use_container_width=True)
                if clear_probe:
                    st.session_state.probe_results = []
                    st.rerun()
        
        if run_probe:
            if not gemini_client.is_connected:
                st.error("❌ Gemini API not connected. Please configure in sidebar.")
            elif not target_url or not questions_data:
                st.error("❌ Please provide target URL and at least one question.")
            else:
                with st.spinner("🔄 Running probes with Gemini API... This may take a few minutes..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    
                    for idx, q_data in enumerate(questions_data):
                        status_text.text(f"Probing fact {idx + 1}/{len(questions_data)}...")
                        
                        result = run_probe_with_paraphrases(
                            gemini_client,
                            q_data['main'],
                            q_data['paraphrases'],
                            target_url,
                            model_choice,
                            temperature,
                            k_runs
                        )
                        result['expected'] = q_data['expected']
                        result['page_name'] = page_name
                        results.append(result)
                        
                        progress_bar.progress((idx + 1) / len(questions_data))
                    
                    st.session_state.probe_results = results
                    status_text.text("")
                    st.success(f"✅ Completed {len(results)} probes!")
                    st.rerun()
        
        # Display Results (same as before - keeping code concise)
        if st.session_state.probe_results:
            st.markdown("---")
            st.markdown("### 📊 Probe Results")
            
            results = st.session_state.probe_results
            avg_consistency = np.mean([r['consistency_score'] for r in results])
            robust_count = sum([r['paraphrase_robust'] for r in results])
            needs_retrieval_count = sum([r['needs_retrieval'] for r in results])
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{len(results)}</h3>
                    <p style="margin:0;">Total Probes</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{avg_consistency:.1f}%</h3>
                    <p style="margin:0;">Avg Consistency</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                card_class = "success-card" if robust_count == len(results) else "warning-card"
                st.markdown(f"""
                <div class="{card_class}">
                    <h3 style="margin:0; font-size:2rem;">{robust_count}/{len(results)}</h3>
                    <p style="margin:0;">Paraphrase Robust</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                card_class = "warning-card" if needs_retrieval_count > 0 else "success-card"
                st.markdown(f"""
                <div class="{card_class}">
                    <h3 style="margin:0; font-size:2rem;">{needs_retrieval_count}</h3>
                    <p style="margin:0;">Need Retrieval</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            fig_consistency = create_consistency_chart(results)
            st.plotly_chart(fig_consistency, use_container_width=True)
            
            st.markdown("#### 📋 Detailed Results")
            
            for idx, r in enumerate(results, 1):
                with st.expander(f"Probe {idx}: {r['original_question'][:60]}...", expanded=(idx==1)):
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    
                    with col_detail1:
                        st.metric("Consistency", f"{r['consistency_score']:.1f}%")
                    with col_detail2:
                        st.metric("Paraphrase Robust", "✓ Yes" if r['paraphrase_robust'] else "✗ No")
                    with col_detail3:
                        st.metric("Needs Retrieval", "✓ Yes" if r['needs_retrieval'] else "✗ No")
                    
                    st.markdown("**Most Common Answer:**")
                    st.info(r.get('most_common_answer', 'N/A'))
                    
                    if r.get('expected'):
                        st.markdown(f"**Expected Answer:** {r['expected']}")
                    
                    st.markdown("**All Responses:**")
                    for q, responses in r['responses'].items():
                        st.markdown(f"*Question variant:* {q[:80]}...")
                        for i, resp in enumerate(responses, 1):
                            st.text(f"  Run {i}: {resp[:150]}{'...' if len(resp) > 150 else ''}")
            
            # Download Options
            st.markdown("#### 💾 Export Results")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                if target_url:
                    report_md = generate_markdown_report(results, target_url)
                    st.download_button(
                        label="📄 Download Report (Markdown)",
                        data=report_md,
                        file_name=f"probe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with col_dl2:
                results_df = pd.DataFrame([{
                    'Question': r['original_question'],
                    'Most Common Answer': r.get('most_common_answer', 'N/A'),
                    'Expected': r.get('expected', 'N/A'),
                    'Consistency': f"{r['consistency_score']:.1f}%",
                    'Robust': r['paraphrase_robust'],
                    'Needs Retrieval': r['needs_retrieval'],
                    'Model': r.get('model_used', 'N/A')
                } for r in results])
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Results (CSV)",
                    data=csv,
                    file_name=f"probe_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ==================== TAB 2: LINK AUTHORITY ANALYSIS (OFFLINE) ====================
    with tab2:
        st.markdown('<h2 class="sub-header">Link Authority Analysis (Offline Mode)</h2>', 
                    unsafe_allow_html=True)
        
        if not st.session_state.similarity_model_loaded:
            st.error("⚠️ Similarity model not loaded. Please load the model in the sidebar first.")
        
        st.markdown("""
        <div class="info-box">
            <strong>🔗 Offline Semantic Similarity:</strong> Analyze the topical alignment between 
            external linking pages and your target pages using local Sentence Transformer models.
            <strong>No external API required!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Pair Check", "Bulk Analysis (CSV Upload)"],
            horizontal=True
        )
        
        if analysis_mode == "Single Pair Check":
            st.markdown("#### 🔍 Single Pair Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                external_url_single = st.text_input(
                    "External URL (Linking Page)",
                    placeholder="https://example.com/casino-reviews",
                    help="The page that links to your content"
                )
            
            with col2:
                target_url_single = st.text_input(
                    "Target URL (Your Page)",
                    placeholder="https://stake.com/casino/games/blackjack",
                    help="Your page being linked to"
                )
            
            if st.button("🚀 Analyze Pair", use_container_width=True, type="primary"):
                if not st.session_state.similarity_model_loaded:
                    st.error("❌ Similarity model not loaded. Please load model in sidebar.")
                elif not external_url_single or not target_url_single:
                    st.error("❌ Please provide both URLs")
                else:
                    with st.spinner("🔄 Analyzing semantic similarity offline..."):
                        similarity_client = st.session_state.similarity_model
                        result = similarity_client.get_similarity(
                            external_url_single,
                            target_url_single
                        )
                        
                        if result:
                            col_r1, col_r2, col_r3 = st.columns(3)
                            
                            with col_r1:
                                st.metric("Similarity Score", f"{result['similarity']:.3f}")
                            
                            with col_r2:
                                bucket, badge_class = get_similarity_score_bucket(result['similarity'])
                                st.markdown(f"""
                                <div class="score-badge {badge_class}">
                                    {bucket}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_r3:
                                if result['similarity'] <= 0.45:
                                    st.markdown('<div class="warning-card">⚠️ OFF-AUTHORITY</div>', 
                                              unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="success-card">✓ ACCEPTABLE</div>', 
                                              unsafe_allow_html=True)
                            
                            st.markdown("#### 📊 Detailed Analysis")
                            st.json(result)
                            
                            if 'single_results' not in st.session_state:
                                st.session_state.single_results = []
                            st.session_state.single_results.append(result)
                            
                            st.success("✅ Analysis complete!")
                        else:
                            st.error("❌ Analysis failed. Check logs for details.")
        
        else:  # Bulk Analysis
            st.markdown("#### 📤 Bulk CSV Upload")
            
            st.markdown("""
            <div class="info-box">
                <strong>CSV Format Required:</strong>
                <ul>
                    <li>Header: <code>external_url,target_url</code></li>
                    <li>Each row = one pair to analyze</li>
                    <li>Full URLs required</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Generate Sample CSV Template"):
                sample_data = pd.DataFrame({
                    'external_url': [
                        'https://en.wikipedia.org/wiki/Casino',
                        'https://en.wikipedia.org/wiki/Blackjack',
                        'https://en.wikipedia.org/wiki/Poker'
                    ],
                    'target_url': [
                        'https://en.wikipedia.org/wiki/Gambling',
                        'https://en.wikipedia.org/wiki/Card_game',
                        'https://en.wikipedia.org/wiki/Texas_hold_em'
                    ]
                })
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download pairs_sample.csv",
                    data=csv,
                    file_name="pairs_sample.csv",
                    mime="text/csv"
                )
            
            uploaded_file = st.file_uploader(
                "Upload pairs.csv",
                type=['csv'],
                help="CSV file with external_url and target_url columns"
            )
            
            if uploaded_file:
                try:
                    pairs_df = pd.read_csv(uploaded_file)
                    
                    if 'external_url' not in pairs_df.columns or 'target_url' not in pairs_df.columns:
                        st.error("❌ CSV must have 'external_url' and 'target_url' columns")
                    else:
                        st.success(f"✅ Loaded {len(pairs_df)} pairs")
                        st.dataframe(pairs_df.head(10), use_container_width=True)
                        
                        if st.button("🚀 Run Bulk Analysis", type="primary", use_container_width=True):
                            if not st.session_state.similarity_model_loaded:
                                st.error("❌ Similarity model not loaded. Please load in sidebar.")
                            else:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                similarity_client = st.session_state.similarity_model
                                results = []
                                
                                for idx, row in pairs_df.iterrows():
                                    status_text.text(f"Processing pair {idx+1}/{len(pairs_df)}...")
                                    
                                    result = similarity_client.get_similarity(
                                        row['external_url'],
                                        row['target_url']
                                    )
                                    
                                    if result:
                                        results.append(result)
                                    else:
                                        logger.warning(f"Failed to process pair {idx+1}")
                                    
                                    progress_bar.progress((idx + 1) / len(pairs_df))
                                    time.sleep(0.1)
                                
                                st.session_state.link_analysis_results = results
                                status_text.text("")
                                st.success(f"✅ Analyzed {len(results)}/{len(pairs_df)} pairs!")
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
                    logger.error(f"CSV reading error: {e}")
        
        # Display Bulk Results
        if st.session_state.link_analysis_results:
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            results_df = pd.DataFrame(st.session_state.link_analysis_results)
            
            avg_similarity = results_df['similarity'].mean()
            off_topic_count = (results_df['interpretation'] == 'Off-Topic').sum()
            on_topic_count = (results_df['interpretation'] == 'On-Topic').sum()
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{len(results_df)}</h3>
                    <p style="margin:0;">Total Pairs</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; font-size:2rem;">{avg_similarity:.3f}</h3>
                    <p style="margin:0;">Avg Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="success-card">
                    <h3 style="margin:0; font-size:2rem;">{on_topic_count}</h3>
                    <p style="margin:0;">On-Topic</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="warning-card">
                    <h3 style="margin:0; font-size:2rem;">{off_topic_count}</h3>
                    <p style="margin:0;">Off-Topic</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig_dist = create_similarity_distribution(results_df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col_v2:
                fig_pie = create_interpretation_pie(results_df)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            if len(results_df) > 2:
                fig_heatmap = create_heatmap_analysis(results_df)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("#### 📋 Detailed Results")
            display_df = results_df[[
                'external_url', 'target_url', 'similarity', 
                'interpretation', 'timestamp'
            ]].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### 💾 Export Results")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download results.csv",
                    data=csv,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_dl2:
                audit_card = results_df['interpretation'].value_counts().reset_index()
                audit_card.columns = ['Interpretation', 'Count']
                audit_csv = audit_card.to_csv(index=False)
                st.download_button(
                    label="📊 Download audit_card.csv",
                    data=audit_csv,
                    file_name=f"audit_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ==================== TAB 3 & 4: Same as before ====================
    with tab3:
        st.markdown('<h2 class="sub-header">Domain Pattern Analysis</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>🎯 Link Farm Detection:</strong> Identify domains with suspicious patterns using offline analysis.
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.link_analysis_results:
            st.info("👆 Run Link Authority Analysis first to see domain patterns")
        else:
            results_df = pd.DataFrame(st.session_state.link_analysis_results)
            
            if st.button("🔄 Analyze Domain Patterns", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing domain patterns..."):
                    domain_summary = analyze_domain_pattern(
                        results_df,
                        min_pairs=min_pairs,
                        off_topic_threshold=off_topic_threshold,
                        avg_score_threshold=avg_score_threshold
                    )
                    st.session_state.domain_summary = domain_summary
                    st.success("✅ Analysis complete!")
            
            if isinstance(st.session_state.domain_summary, pd.DataFrame) and \
               len(st.session_state.domain_summary) > 0:
                
                domain_summary = st.session_state.domain_summary
                
                total_domains = len(domain_summary)
                flagged_domains = domain_summary['flagged'].sum()
                worst_score = domain_summary['avg_score'].min()
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; font-size:2rem;">{total_domains}</h3>
                        <p style="margin:0;">Total Domains</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    card_class = "warning-card" if flagged_domains > 0 else "success-card"
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3 style="margin:0; font-size:2rem;">{flagged_domains}</h3>
                        <p style="margin:0;">Flagged Domains</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown(f"""
                    <div class="info-card">
                        <h3 style="margin:0; font-size:2rem;">{worst_score:.3f}</h3>
                        <p style="margin:0;">Worst Avg Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    flag_rate = (flagged_domains / total_domains * 100) if total_domains > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; font-size:2rem;">{flag_rate:.1f}%</h3>
                        <p style="margin:0;">Flag Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                fig_domain = create_domain_comparison(domain_summary)
                st.plotly_chart(fig_domain, use_container_width=True)
                
                flagged_df = domain_summary[domain_summary['flagged'] == True]
                
                if len(flagged_df) > 0:
                    st.markdown("### 🚨 Suspect Link Farms")
                    
                    for _, row in flagged_df.iterrows():
                        with st.expander(f"⚠️ {row['external_domain']} - FLAGGED", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Avg Score", f"{row['avg_score']:.3f}")
                            with col2:
                                st.metric("Off-Topic Rate", f"{row['off_topic_rate']:.1%}")
                            with col3:
                                st.metric("Total Pairs", int(row['pair_count']))
                            with col4:
                                st.metric("Verdict", row['verdict'])
                            
                            domain_urls = results_df[
                                results_df['external_url'].str.contains(row['external_domain'], regex=False)
                            ][['external_url', 'target_url', 'similarity', 'interpretation']].head(5)
                            
                            st.markdown("**Example Links:**")
                            st.dataframe(domain_urls, use_container_width=True, hide_index=True)
                else:
                    st.markdown("""
                    <div class="success-card">
                        <h3>✅ No Suspect Domains Found</h3>
                        <p>All analyzed domains meet quality thresholds</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📊 All Domains Summary")
                
                display_summary = domain_summary.copy()
                display_summary['avg_score'] = display_summary['avg_score'].round(3)
                display_summary['off_topic_rate'] = (display_summary['off_topic_rate'] * 100).round(1)
                display_summary = display_summary.rename(columns={
                    'external_domain': 'Domain',
                    'avg_score': 'Avg Score',
                    'min_score': 'Min Score',
                    'max_score': 'Max Score',
                    'pair_count': 'Pairs',
                    'off_topic_count': 'Off-Topic Count',
                    'off_topic_rate': 'Off-Topic %',
                    'flagged': 'Flagged',
                    'verdict': 'Verdict'
                })
                
                st.dataframe(display_summary, use_container_width=True, hide_index=True)
                
                st.markdown("#### 💾 Export Domain Analysis")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    csv = domain_summary.to_csv(index=False)
                    st.download_button(
                        label="📊 Download domain_summary_cmseo.csv",
                        data=csv,
                        file_name=f"domain_summary_cmseo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_dl2:
                    findings_md = generate_cmseo_findings(domain_summary)
                    st.download_button(
                        label="📄 Download cmseo_findings.md",
                        data=findings_md,
                        file_name=f"cmseo_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
    
    with tab4:
        st.markdown('<h2 class="sub-header">Analytics Dashboard</h2>', 
                    unsafe_allow_html=True)
        
        has_probe_data = len(st.session_state.probe_results) > 0
        has_link_data = len(st.session_state.link_analysis_results) > 0
        
        if not has_probe_data and not has_link_data:
            st.info("📊 No data available yet. Run analyses in other tabs to see comprehensive analytics.")
        else:
            st.markdown("### 📈 Combined Overview")
            
            overview_cols = st.columns(3)
            
            with overview_cols[0]:
                if has_probe_data:
                    avg_cons = np.mean([r['consistency_score'] for r in st.session_state.probe_results])
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Model Probing</h4>
                        <p><strong>{len(st.session_state.probe_results)}</strong> probes completed</p>
                        <p>Avg consistency: <strong>{avg_cons:.1f}%</strong></p>
                        <p>Model: <strong>{st.session_state.probe_results[0].get('model_used', 'N/A')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with overview_cols[1]:
                if has_link_data:
                    results_df = pd.DataFrame(st.session_state.link_analysis_results)
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Link Analysis (Offline)</h4>
                        <p><strong>{len(results_df)}</strong> pairs analyzed</p>
                        <p>Avg similarity: <strong>{results_df['similarity'].mean():.3f}</strong></p>
                        <p>On-topic: <strong>{(results_df['interpretation'] == 'On-Topic').sum()}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with overview_cols[2]:
                if isinstance(st.session_state.domain_summary, pd.DataFrame):
                    domain_summary = st.session_state.domain_summary
                    flagged = domain_summary['flagged'].sum()
                    card_class = "warning-card" if flagged > 0 else "success-card"
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>Domain Patterns</h4>
                        <p><strong>{len(domain_summary)}</strong> domains analyzed</p>
                        <p><strong>{flagged}</strong> flagged as suspect</p>
                    </div>
                    """, unsafe_allow_html=True)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
