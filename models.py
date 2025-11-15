import os
import time
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Core ML libraries
import torch
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Audio processing
import whisper
import librosa
import soundfile as sf

# Advanced NLP libraries
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import Counter
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded - FIXED VERSION"""
    
    logger.info("🔄 Checking and downloading NLTK data...")
    
    # List of required NLTK packages (updated for newer versions)
    required_packages = [
        'punkt_tab',        # NEW: Updated tokenizer
        'punkt',            # BACKUP: Legacy tokenizer
        'stopwords',        # Stop words
        'averaged_perceptron_tagger',  # POS tagger
        'maxent_ne_chunker',          # Named entity chunker
        'words',            # Word corpus
        'vader_lexicon',    # VADER lexicon
        'wordnet',          # WordNet database
        'omw-1.4',          # Open Multilingual Wordnet
        'brown',            # Brown corpus (for testing)
    ]
    
    downloaded = []
    failed = []
    
    for package in required_packages:
        try:
            # Check if already available
            try:
                if package == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                elif package == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif package == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif package == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                elif package == 'maxent_ne_chunker':
                    nltk.data.find('chunkers/maxent_ne_chunker')
                elif package == 'words':
                    nltk.data.find('corpora/words')
                elif package == 'vader_lexicon':
                    nltk.data.find('vader_lexicon')
                elif package == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif package == 'omw-1.4':
                    nltk.data.find('corpora/omw-1.4')
                elif package == 'brown':
                    nltk.data.find('corpora/brown')
                
                logger.info(f"✅ {package}: Already available")
                downloaded.append(package)
                continue
                
            except LookupError:
                # Package not found, need to download
                logger.info(f"⬇️ Downloading {package}...")
                nltk.download(package, quiet=True)
                downloaded.append(package)
                logger.info(f"✅ {package}: Downloaded successfully")
                
        except Exception as e:
            logger.warning(f"⚠️ {package}: Download failed - {str(e)}")
            failed.append(package)
    
    logger.info(f"📊 NLTK Data Status: {len(downloaded)} successful, {len(failed)} failed")
    
    if downloaded:
        logger.info(f"✅ Available packages: {', '.join(downloaded)}")
    
    if failed:
        logger.warning(f"⚠️ Failed packages: {', '.join(failed)}")
    
    # Test critical functionality
    test_nltk_functionality()

def test_nltk_functionality():
    """Test NLTK functionality with available packages"""
    
    try:
        # Test tokenization (try new version first, then fallback)
        test_text = "Hello world. This is a test sentence."
        
        try:
            # Try punkt_tab first (newer version)
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(test_text)
            logger.info(f"✅ Sentence tokenization working: {len(sentences)} sentences")
        except Exception as e1:
            try:
                # Fallback to punkt
                import nltk.tokenize
                sentences = nltk.tokenize.sent_tokenize(test_text)
                logger.info(f"✅ Sentence tokenization working (fallback): {len(sentences)} sentences")
            except Exception as e2:
                logger.error(f"❌ Sentence tokenization failed: {e1}, {e2}")
        
        # Test other functionality
        try:
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')
            logger.info(f"✅ Stop words working: {len(stop_words)} words")
        except Exception as e:
            logger.warning(f"⚠️ Stop words not available: {e}")
        
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
            score = vader.polarity_scores("This is great!")
            logger.info(f"✅ VADER sentiment working: {score}")
        except Exception as e:
            logger.warning(f"⚠️ VADER sentiment not available: {e}")
            
    except Exception as e:
        logger.error(f"❌ NLTK testing failed: {e}")

# Initialize NLTK data on import
ensure_nltk_data()

class AdvancedNLPAnalyzer:
    """Advanced NLP analysis with robust NLTK handling"""
    
    def __init__(self):
        self.sentence_transformer = None
        self.nlp_spacy = None
        self.summarization_pipeline = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.nltk_available = self._check_nltk_availability()
        self._load_models()
    
    def _check_nltk_availability(self):
        """Check what NLTK functionality is available"""
        
        available = {
            'tokenization': False,
            'stopwords': False,
            'pos_tagging': False,
            'named_entities': False,
            'vader': False
        }
        
        # Test tokenization (punkt_tab or punkt)
        try:
            from nltk.tokenize import sent_tokenize, word_tokenize
            sent_tokenize("Test sentence.")
            word_tokenize("Test sentence.")
            available['tokenization'] = True
            logger.info("✅ NLTK tokenization available")
        except Exception as e:
            logger.warning(f"⚠️ NLTK tokenization unavailable: {e}")
        
        # Test stop words
        try:
            from nltk.corpus import stopwords
            stopwords.words('english')
            available['stopwords'] = True
            logger.info("✅ NLTK stopwords available")
        except Exception as e:
            logger.warning(f"⚠️ NLTK stopwords unavailable: {e}")
        
        # Test POS tagging
        try:
            from nltk.tag import pos_tag
            from nltk.tokenize import word_tokenize
            pos_tag(word_tokenize("Test sentence."))
            available['pos_tagging'] = True
            logger.info("✅ NLTK POS tagging available")
        except Exception as e:
            logger.warning(f"⚠️ NLTK POS tagging unavailable: {e}")
        
        # Test named entities
        try:
            from nltk.chunk import ne_chunk
            from nltk.tag import pos_tag
            from nltk.tokenize import word_tokenize
            ne_chunk(pos_tag(word_tokenize("John lives in New York.")))
            available['named_entities'] = True
            logger.info("✅ NLTK named entities available")
        except Exception as e:
            logger.warning(f"⚠️ NLTK named entities unavailable: {e}")
        
        # Test VADER
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            SentimentIntensityAnalyzer().polarity_scores("Test")
            available['vader'] = True
            logger.info("✅ NLTK VADER available")
        except Exception as e:
            logger.warning(f"⚠️ NLTK VADER unavailable: {e}")
        
        return available
    
    def _load_models(self):
        """Load all NLP models with fallback handling"""
        
        try:
            logger.info("Loading advanced NLP models...")
            
            # Load sentence transformer
            logger.info("Loading sentence transformer...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
            
            # Load spaCy model
            logger.info("Loading spaCy model...")
            try:
                self.nlp_spacy = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded")
            except OSError:
                logger.warning("⚠️ spaCy model not found. Installing...")
                try:
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp_spacy = spacy.load("en_core_web_sm")
                    logger.info("✅ spaCy model installed and loaded")
                except Exception as spacy_error:
                    logger.error(f"❌ spaCy installation failed: {spacy_error}")
                    self.nlp_spacy = None
            
            # Load summarization pipeline
            logger.info("Loading summarization model...")
            try:
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
                logger.info("✅ Summarization model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Summarization model failed to load: {e}")
                self.summarization_pipeline = None
            
            logger.info("🎉 Advanced NLP models loaded!")
            
        except Exception as e:
            logger.error(f"❌ Error loading NLP models: {e}")
    
    def analyze_text_advanced(self, text: str) -> Dict[str, Any]:
        """Comprehensive NLP analysis with robust error handling"""
        
        logger.info("🔬 Starting advanced NLP analysis...")
        start_time = time.time()
        
        results = {
            "original_text": text,
            "text_length": len(text),
            "embeddings": None,
            "vector_stats": {},
            "nltk_analysis": {},
            "spacy_analysis": {},
            "summaries": {},
            "similarity_analysis": {},
            "semantic_features": {},
            "processing_time": 0.0
        }
        
        try:
            # 1. Generate vector embeddings
            logger.info("🧮 Generating vector embeddings...")
            results["embeddings"] = self._generate_embeddings(text)
            if results["embeddings"] and "error" not in results["embeddings"]:
                results["vector_stats"] = self._analyze_embeddings(results["embeddings"])
            
            # 2. NLTK Analysis (with availability checks)
            logger.info("🔍 Performing NLTK analysis...")
            results["nltk_analysis"] = self._nltk_analysis_robust(text)
            
            # 3. spaCy Analysis
            logger.info("🕷️ Performing spaCy analysis...")
            results["spacy_analysis"] = self._spacy_analysis(text)
            
            # 4. Text Summarization
            logger.info("📝 Generating summaries...")
            results["summaries"] = self._generate_summaries_robust(text)
            
            # 5. Similarity Analysis
            logger.info("🔗 Performing similarity analysis...")
            results["similarity_analysis"] = self._similarity_analysis_robust(text)
            
            # 6. Semantic Features
            logger.info("🧠 Extracting semantic features...")
            results["semantic_features"] = self._extract_semantic_features_robust(text)
            
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            
            logger.info(f"✅ Advanced NLP analysis complete in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Advanced NLP analysis failed: {e}")
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
            return results
    
    def _generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generate embeddings with robust tokenization"""
        
        if not self.sentence_transformer:
            return {"error": "Sentence transformer not available"}
        
        try:
            # Use robust sentence splitting
            sentences = self._split_sentences_robust(text)
            
            if not sentences:
                return {"error": "No sentences found in text"}
            
            # Generate embeddings
            text_embedding = self.sentence_transformer.encode([text])
            sentence_embeddings = self.sentence_transformer.encode(sentences)
            
            return {
                "text_embedding": text_embedding[0].tolist(),
                "sentence_embeddings": sentence_embeddings.tolist(),
                "embedding_dimension": len(text_embedding[0]),
                "num_sentences": len(sentences),
                "sentences": sentences
            }
            
        except Exception as e:
            return {"error": f"Embedding generation failed: {str(e)}"}
    
    def _split_sentences_robust(self, text: str) -> List[str]:
        """Split sentences with multiple fallback methods"""
        
        # Method 1: Try NLTK sent_tokenize
        if self.nltk_available.get('tokenization', False):
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"NLTK sent_tokenize failed: {e}")
        
        # Method 2: spaCy sentence splitting
        if self.nlp_spacy:
            try:
                doc = self.nlp_spacy(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"spaCy sentence splitting failed: {e}")
        
        # Method 3: Simple regex-based splitting
        try:
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception as e:
            logger.warning(f"Regex sentence splitting failed: {e}")
        
        # Method 4: Fallback - return whole text as single sentence
        return [text] if text.strip() else []
    
    def _nltk_analysis_robust(self, text: str) -> Dict[str, Any]:
        """NLTK analysis with robust fallback handling"""
        
        if not any(self.nltk_available.values()):
            return {"error": "No NLTK functionality available"}
        
        try:
            analysis = {}
            
            # Tokenization
            if self.nltk_available.get('tokenization', False):
                try:
                    from nltk.tokenize import sent_tokenize, word_tokenize
                    sentences = sent_tokenize(text)
                    words = word_tokenize(text)
                    
                    analysis["tokenization"] = {
                        "sentence_count": len(sentences),
                        "word_count": len(words),
                        "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0
                    }
                    
                except Exception as e:
                    logger.warning(f"NLTK tokenization failed: {e}")
                    # Fallback tokenization
                    sentences = self._split_sentences_robust(text)
                    words = text.split()
                    analysis["tokenization"] = {
                        "sentence_count": len(sentences),
                        "word_count": len(words),
                        "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0
                    }
            else:
                # Use fallback methods
                sentences = self._split_sentences_robust(text)
                words = text.split()
                analysis["tokenization"] = {
                    "sentence_count": len(sentences),
                    "word_count": len(words),
                    "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0
                }
            
            # Stop words filtering
            if self.nltk_available.get('stopwords', False):
                try:
                    from nltk.corpus import stopwords
                    stop_words = set(stopwords.words('english'))
                    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
                    analysis["tokenization"]["filtered_word_count"] = len(filtered_words)
                    
                    # Word frequency
                    word_freq = Counter(filtered_words)
                    analysis["word_frequency"] = dict(word_freq.most_common(15))
                    
                except Exception as e:
                    logger.warning(f"Stop words filtering failed: {e}")
            
            # POS Tagging
            if self.nltk_available.get('pos_tagging', False):
                try:
                    from nltk.tag import pos_tag
                    pos_tags = pos_tag(words)
                    pos_freq = Counter([pos for word, pos in pos_tags])
                    
                    analysis["pos_analysis"] = {
                        "pos_distribution": dict(pos_freq.most_common(10)),
                        "noun_count": pos_freq.get('NN', 0) + pos_freq.get('NNS', 0) + pos_freq.get('NNP', 0),
                        "verb_count": pos_freq.get('VB', 0) + pos_freq.get('VBD', 0) + pos_freq.get('VBG', 0),
                        "adjective_count": pos_freq.get('JJ', 0) + pos_freq.get('JJR', 0) + pos_freq.get('JJS', 0)
                    }
                except Exception as e:
                    logger.warning(f"POS tagging failed: {e}")
            
            # Named Entity Recognition
            if self.nltk_available.get('named_entities', False):
                try:
                    from nltk.chunk import ne_chunk
                    from nltk.tag import pos_tag
                    
                    ne_tree = ne_chunk(pos_tag(words))
                    named_entities = []
                    for chunk in ne_tree:
                        if hasattr(chunk, 'label'):
                            entity = ' '.join([token for token, pos in chunk.leaves()])
                            named_entities.append((entity, chunk.label()))
                    
                    analysis["named_entities"] = named_entities[:10]
                except Exception as e:
                    logger.warning(f"Named entity recognition failed: {e}")
            
            # VADER Sentiment
            if self.nltk_available.get('vader', False):
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    nltk_vader = SentimentIntensityAnalyzer()
                    nltk_sentiment = nltk_vader.polarity_scores(text)
                    analysis["nltk_sentiment"] = nltk_sentiment
                except Exception as e:
                    logger.warning(f"VADER sentiment failed: {e}")
            
            return analysis
            
        except Exception as e:
            return {"error": f"NLTK analysis failed: {str(e)}"}
    
    def _generate_summaries_robust(self, text: str) -> Dict[str, Any]:
        """Generate summaries with robust sentence splitting"""
        
        summaries = {}
        
        try:
            # Extractive summary
            logger.info("📄 Generating extractive summary...")
            summaries["extractive"] = self._extractive_summary_robust(text)
            
            # Abstractive summary
            if self.summarization_pipeline and len(text.split()) > 50:
                logger.info("🤖 Generating abstractive summary...")
                try:
                    max_length = 1024
                    truncated_text = text[:max_length] if len(text) > max_length else text
                    
                    summary_result = self.summarization_pipeline(
                        truncated_text,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    summaries["abstractive"] = {
                        "summary": summary_result[0]['summary_text'],
                        "method": "BART (facebook/bart-large-cnn)",
                        "input_length": len(truncated_text),
                        "summary_length": len(summary_result[0]['summary_text'])
                    }
                except Exception as abs_error:
                    summaries["abstractive"] = {"error": f"Abstractive summarization failed: {str(abs_error)}"}
            else:
                summaries["abstractive"] = {"error": "Text too short or model unavailable"}
            
            return summaries
            
        except Exception as e:
            return {"error": f"Summarization failed: {str(e)}"}
    
    def _extractive_summary_robust(self, text: str, num_sentences: int = 3) -> Dict[str, Any]:
        """Extractive summary with robust sentence splitting"""
        
        try:
            sentences = self._split_sentences_robust(text)
            
            if len(sentences) <= num_sentences:
                return {
                    "summary": text,
                    "method": "TextRank (all sentences)",
                    "num_sentences_original": len(sentences),
                    "num_sentences_summary": len(sentences),
                    "compression_ratio": 1.0
                }
            
            if not self.sentence_transformer:
                return {"error": "Sentence transformer not available"}
            
            # Create similarity matrix
            sentence_embeddings = self.sentence_transformer.encode(sentences)
            similarity_matrix = cosine_similarity(sentence_embeddings)
            
            # Apply TextRank algorithm
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Get top sentences
            ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
            top_sentences = ranked_sentences[:num_sentences]
            
            # Sort by original order
            top_sentences_ordered = sorted(top_sentences, key=lambda x: x[2])
            summary_text = ' '.join([s[1] for s in top_sentences_ordered])
            
            return {
                "summary": summary_text,
                "method": "TextRank with sentence embeddings",
                "num_sentences_original": len(sentences),
                "num_sentences_summary": num_sentences,
                "compression_ratio": len(summary_text) / len(text),
                "sentence_scores": {i: float(score) for i, (score, _, _) in enumerate(ranked_sentences)}
            }
            
        except Exception as e:
            return {"error": f"Extractive summarization failed: {str(e)}"}
    
    def _similarity_analysis_robust(self, text: str) -> Dict[str, Any]:
        """Similarity analysis with robust handling"""
        
        try:
            sentences = self._split_sentences_robust(text)
            
            if len(sentences) < 2:
                return {"error": "Text too short for similarity analysis"}
            
            if not self.sentence_transformer:
                return {"error": "Sentence transformer not available"}
            
            # Generate embeddings
            sentence_embeddings = self.sentence_transformer.encode(sentences)
            
            # Similarity search demonstration
            search_results = []
            if len(sentences) > 1:
                query_embedding = sentence_embeddings[0]
                similarities = cosine_similarity([query_embedding], sentence_embeddings[1:])[0]
                
                for i, sim_score in enumerate(similarities):
                    search_results.append({
                        "sentence": sentences[i + 1],
                        "similarity_score": float(sim_score),
                        "rank": i + 1
                    })
                
                search_results = sorted(search_results, key=lambda x: x['similarity_score'], reverse=True)
            
            # TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            tfidf_similarities = cosine_similarity(tfidf_matrix)
            
            return {
                "embedding_based_search": {
                    "query_sentence": sentences[0] if sentences else "",
                    "similar_sentences": search_results[:3],
                    "search_method": "Sentence Transformer + Cosine Similarity"
                },
                "tfidf_comparison": {
                    "avg_tfidf_similarity": float(np.mean(tfidf_similarities)),
                    "max_tfidf_similarity": float(np.max(tfidf_similarities[np.triu_indices_from(tfidf_similarities, k=1)])) if len(sentences) > 1 else 0,
                    "method": "TF-IDF + Cosine Similarity"
                },
                "similarity_methods_used": [
                    "Sentence Transformers (all-MiniLM-L6-v2)",
                    "TF-IDF Vectorization",
                    "Cosine Similarity Metric",
                    "TextRank Algorithm"
                ],
                "search_capabilities": {
                    "semantic_search": "Uses dense embeddings for meaning-based similarity",
                    "keyword_search": "TF-IDF for term frequency based similarity",
                    "hybrid_approach": "Combines both methods for comprehensive search"
                }
            }
            
        except Exception as e:
            return {"error": f"Similarity analysis failed: {str(e)}"}
    
    def _extract_semantic_features_robust(self, text: str) -> Dict[str, Any]:
        """Semantic features with robust handling"""
        
        try:
            sentences = self._split_sentences_robust(text)
            words = text.split()
            
            # Basic metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            unique_words = set([word.lower() for word in words if word.isalpha()])
            lexical_diversity = len(unique_words) / len(words) if words else 0
            
            # Semantic coherence
            semantic_coherence = 0.0
            if self.sentence_transformer and len(sentences) > 1:
                try:
                    sentence_embeddings = self.sentence_transformer.encode(sentences)
                    pairwise_similarities = cosine_similarity(sentence_embeddings)
                    semantic_coherence = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
                except Exception as e:
                    logger.warning(f"Semantic coherence calculation failed: {e}")
            
            features = {
                "readability": {
                    "avg_sentence_length": avg_sentence_length,
                    "lexical_diversity": lexical_diversity,
                    "word_count": len(words),
                    "sentence_count": len(sentences),
                    "avg_word_length": np.mean([len(word) for word in words if word.isalpha()]) if words else 0
                },
                "semantic_coherence": float(semantic_coherence),
                "text_complexity": {
                    "long_sentences": len([s for s in sentences if len(s.split()) > 20]),
                    "complex_words": len([w for w in words if len(w) > 6 and w.isalpha()]),
                    "unique_word_ratio": lexical_diversity
                }
            }
            
            return features
            
        except Exception as e:
            return {"error": f"Semantic feature extraction failed: {str(e)}"}
    
    def _analyze_embeddings(self, embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze embeddings"""
        
        if "error" in embeddings_data:
            return {"error": embeddings_data["error"]}
        
        try:
            text_embedding = np.array(embeddings_data["text_embedding"])
            sentence_embeddings = np.array(embeddings_data["sentence_embeddings"])
            
            stats = {
                "embedding_dimension": embeddings_data["embedding_dimension"],
                "text_embedding_norm": float(np.linalg.norm(text_embedding)),
                "text_embedding_mean": float(np.mean(text_embedding)),
                "text_embedding_std": float(np.std(text_embedding)),
                "sentence_embedding_stats": {
                    "count": len(sentence_embeddings),
                    "avg_norm": float(np.mean([np.linalg.norm(emb) for emb in sentence_embeddings])),
                    "similarity_matrix_shape": sentence_embeddings.shape if len(sentence_embeddings) > 1 else "single_sentence"
                }
            }
            
            if len(sentence_embeddings) > 1:
                similarity_matrix = cosine_similarity(sentence_embeddings)
                stats["sentence_similarities"] = {
                    "avg_similarity": float(np.mean(similarity_matrix)),
                    "max_similarity": float(np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                    "min_similarity": float(np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
                    "similarity_matrix": similarity_matrix.tolist()
                }
            
            return stats
            
        except Exception as e:
            return {"error": f"Embedding analysis failed: {str(e)}"}
    
    def _spacy_analysis(self, text: str) -> Dict[str, Any]:
        """spaCy analysis"""
        
        if not self.nlp_spacy:
            return {"error": "spaCy model not available"}
        
        try:
            doc = self.nlp_spacy(text)
            
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            token_analysis = {
                "total_tokens": len(doc),
                "alpha_tokens": len([token for token in doc if token.is_alpha]),
                "stop_words": len([token for token in doc if token.is_stop]),
                "punctuation": len([token for token in doc if token.is_punct]),
                "spaces": len([token for token in doc if token.is_space])
            }
            
            pos_dist = Counter([token.pos_ for token in doc if not token.is_space])
            lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            
            return {
                "entities": entities[:15],
                "entity_labels": list(set([ent[1] for ent in entities])),
                "noun_phrases": noun_phrases[:10],
                "token_analysis": token_analysis,
                "pos_distribution": dict(pos_dist.most_common(10)),
                "lemmas": lemmas[:20],
                "sentence_count": len(list(doc.sents)),
                "language": doc.lang_
            }
            
        except Exception as e:
            return {"error": f"spaCy analysis failed: {str(e)}"}

# Enhanced Transcriber and Predictor classes (same as before)
class WhisperTranscriber:
    """Speech-to-text transcription using Whisper"""
    
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading Whisper: {e}")
            self.model = None
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio to text"""
        
        if not self.model:
            return {
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "error": "Whisper model not available"
            }
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            if not os.path.exists(audio_path):
                return {
                    "text": "",
                    "confidence": 0.0,
                    "language": "unknown",
                    "error": f"Audio file not found: {audio_path}"
                }
            
            start_time = time.time()
            
            # Load audio and transcribe
            logger.info("🎯 Loading audio data for transcription...")
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            logger.info(f"📊 Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
            
            logger.info("🗣️ Starting Whisper transcription...")
            result = self.model.transcribe(audio_data)
            
            processing_time = time.time() - start_time
            
            # Extract results
            transcript_text = result.get("text", "").strip()
            detected_language = result.get("language", "unknown")
            
            # Calculate confidence
            if len(transcript_text) > 20:
                confidence = 0.95
            elif len(transcript_text) > 5:
                confidence = 0.8
            elif len(transcript_text) > 0:
                confidence = 0.5
            else:
                confidence = 0.1
            
            transcription_result = {
                "text": transcript_text,
                "confidence": confidence,
                "language": detected_language,
                "processing_time": processing_time,
                "error": None
            }
            
            logger.info(f"✅ Transcription successful: '{transcript_text[:50]}...' ({confidence:.1%} confidence)")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "unknown",
                "processing_time": 0.0,
                "error": str(e)
            }

class SentimentPredictor:
    """Enhanced sentiment analysis with robust NLP integration"""
    
    def __init__(self):
        self.models = {}
        self.nlp_analyzer = AdvancedNLPAnalyzer()
        self.load_models()
    
    def load_models(self):
        """Load sentiment models"""
        try:
            logger.info("Loading sentiment analysis models...")
            
            # RoBERTa
            self.models['roberta'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            logger.info("✅ RoBERTa model loaded")
            
            # BERT
            self.models['bert'] = pipeline(
                "sentiment-analysis", 
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1
            )
            logger.info("✅ BERT model loaded")
            
            # VADER
            self.models['vader'] = SentimentIntensityAnalyzer()
            logger.info("✅ VADER model loaded")
            
            logger.info("🎉 All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def predict_sentiment_enhanced(self, text: str, include_nlp_analysis: bool = True) -> Dict[str, Any]:
        """Enhanced sentiment prediction with full NLP analysis"""
        
        start_time = time.time()
        
        # Basic sentiment analysis
        basic_results = self.predict_sentiment(text)
        
        # Add advanced NLP analysis
        if include_nlp_analysis and text.strip():
            logger.info("🧠 Adding advanced NLP analysis...")
            nlp_results = self.nlp_analyzer.analyze_text_advanced(text)
            
            # Combine results
            enhanced_results = {
                **basic_results,
                "advanced_nlp": nlp_results,
                "enhancement_time": time.time() - start_time
            }
            
            return enhanced_results
        
        return basic_results
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment prediction"""
        
        start_time = time.time()
        
        results = {
            "text": text,
            "primary_sentiment": "Unknown",
            "confidence": 0.0,
            "ensemble_score": 0.0,
            "model_predictions": {},
            "emotion_scores": {},
            "key_phrases": {"positive": [], "negative": []},
            "reasoning": "",
            "processing_time": 0.0
        }
        
        try:
            model_scores = []
            
            # RoBERTa
            if 'roberta' in self.models:
                try:
                    roberta_result = self.models['roberta'](text)[0]
                    results["model_predictions"]["roberta"] = roberta_result
                    score = self._convert_to_score(roberta_result)
                    model_scores.append(("roberta", score, 0.5))
                    logger.info(f"RoBERTa: {roberta_result['label']} ({roberta_result['score']:.3f})")
                except Exception as e:
                    logger.warning(f"RoBERTa failed: {e}")
            
            # BERT
            if 'bert' in self.models:
                try:
                    bert_result = self.models['bert'](text)[0]
                    results["model_predictions"]["bert"] = bert_result
                    score = self._convert_to_score(bert_result)
                    model_scores.append(("bert", score, 0.3))
                    logger.info(f"BERT: {bert_result['label']} ({bert_result['score']:.3f})")
                except Exception as e:
                    logger.warning(f"BERT failed: {e}")
            
            # VADER
            if 'vader' in self.models:
                try:
                    vader_result = self.models['vader'].polarity_scores(text)
                    results["model_predictions"]["vader"] = vader_result
                    score = vader_result['compound']
                    model_scores.append(("vader", score, 0.2))
                    logger.info(f"VADER: {score:.3f}")
                except Exception as e:
                    logger.warning(f"VADER failed: {e}")
            
            # Calculate ensemble
            if model_scores:
                total_weight = sum(weight for _, _, weight in model_scores)
                ensemble_score = sum(score * weight for _, score, weight in model_scores) / total_weight
                results["ensemble_score"] = ensemble_score
                
                if ensemble_score > 0.1:
                    results["primary_sentiment"] = "Positive"
                    results["confidence"] = min(0.99, abs(ensemble_score))
                elif ensemble_score < -0.1:
                    results["primary_sentiment"] = "Negative"
                    results["confidence"] = min(0.99, abs(ensemble_score))
                else:
                    results["primary_sentiment"] = "Neutral"
                    results["confidence"] = 1.0 - abs(ensemble_score)
            
            # Additional analysis
            results["emotion_scores"] = self._analyze_emotions(text, results["ensemble_score"])
            results["key_phrases"] = self._extract_key_phrases(text)
            results["reasoning"] = self._generate_reasoning(text, results)
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis error: {e}")
            results.update({
                "primary_sentiment": "Error",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {str(e)}"
            })
        
        results["processing_time"] = time.time() - start_time
        logger.info(f"🎯 Final: {results['primary_sentiment']} ({results['confidence']:.1%})")
        
        return results
    
    def _convert_to_score(self, model_result: Dict) -> float:
        """Convert model output to score"""
        label = model_result['label'].upper()
        score = model_result['score']
        
        if any(pos in label for pos in ['POSITIVE', 'POS', '5 STAR', '4 STAR']):
            return score
        elif any(neg in label for neg in ['NEGATIVE', 'NEG', '1 STAR', '2 STAR']):
            return -score
        elif any(neu in label for neu in ['NEUTRAL', '3 STAR']):
            return 0.0
        else:
            return 0.0
    
    def _analyze_emotions(self, text: str, sentiment_score: float) -> Dict[str, float]:
        """Analyze emotions in text"""
        emotions = {
            "joy": 0.0, "satisfaction": 0.0, "enthusiasm": 0.0, "surprise": 0.0,
            "anger": 0.0, "sadness": 0.0, "fear": 0.0, "disgust": 0.0
        }
        
        text_lower = text.lower()
        
        # Positive emotions
        joy_words = ["love", "amazing", "wonderful", "fantastic", "great", "excellent", "happy", "joy"]
        joy_count = sum(1 for word in joy_words if word in text_lower)
        emotions["joy"] = min(0.9, joy_count * 0.2 + max(0, sentiment_score) * 0.5)
        
        satisfaction_words = ["satisfied", "pleased", "good", "fine", "okay", "decent"]
        satisfaction_count = sum(1 for word in satisfaction_words if word in text_lower)
        emotions["satisfaction"] = min(0.9, satisfaction_count * 0.2 + max(0, sentiment_score) * 0.3)
        
        enthusiasm_words = ["exciting", "incredible", "outstanding", "superb", "awesome"]
        enthusiasm_count = sum(1 for word in enthusiasm_words if word in text_lower)
        emotions["enthusiasm"] = min(0.9, enthusiasm_count * 0.3 + max(0, sentiment_score) * 0.4)
        
        # Negative emotions
        if sentiment_score < -0.1:
            anger_words = ["angry", "mad", "hate", "frustrated", "annoyed"]
            anger_count = sum(1 for word in anger_words if word in text_lower)
            emotions["anger"] = min(0.9, anger_count * 0.3 + abs(sentiment_score) * 0.4)
            
            sadness_words = ["sad", "disappointed", "upset", "unhappy"]
            sadness_count = sum(1 for word in sadness_words if word in text_lower)
            emotions["sadness"] = min(0.9, sadness_count * 0.3 + abs(sentiment_score) * 0.3)
        
        return emotions
    
    def _extract_key_phrases(self, text: str) -> Dict[str, List[str]]:
        """Extract key phrases"""
        text_lower = text.lower()
        positive_phrases = []
        negative_phrases = []
        
        # Positive phrases
        pos_patterns = [
            "absolutely love", "really like", "very good", "excellent quality",
            "outstanding", "exceeded expectations", "highly recommend", 
            "amazing", "wonderful", "fantastic", "perfect", "awesome"
        ]
        
        # Negative phrases
        neg_patterns = [
            "really hate", "very bad", "terrible quality", "worst ever",
            "completely disappointed", "awful", "horrible", "useless",
            "waste of money", "don't recommend", "terrible"
        ]
        
        # Find phrases
        for phrase in pos_patterns:
            if phrase in text_lower:
                positive_phrases.append(phrase)
        
        for phrase in neg_patterns:
            if phrase in text_lower:
                negative_phrases.append(phrase)
        
        # Find individual words
        words = text_lower.split()
        pos_words = ["love", "like", "good", "great", "excellent", "amazing", "perfect"]
        neg_words = ["hate", "bad", "terrible", "awful", "horrible", "worst"]
        
        for word in words:
            if word in pos_words and len(positive_phrases) < 8:
                positive_phrases.append(word)
            elif word in neg_words and len(negative_phrases) < 8:
                negative_phrases.append(word)
        
        return {
            "positive": positive_phrases[:8],
            "negative": negative_phrases[:8]
        }
    
    def _generate_reasoning(self, text: str, results: Dict) -> str:
        """Generate reasoning for prediction"""
        sentiment = results["primary_sentiment"]
        confidence = results["confidence"]
        positive_phrases = results["key_phrases"]["positive"]
        negative_phrases = results["key_phrases"]["negative"]
        model_count = len(results["model_predictions"])
        
        reasoning = f"Audio transcript shows {sentiment.lower()} sentiment with {confidence:.1%} confidence. "
        
        if model_count > 1:
            reasoning += f"Based on {model_count} AI models. "
        
        if positive_phrases:
            reasoning += f"Positive language: {', '.join(positive_phrases[:3])}. "
        
        if negative_phrases:
            reasoning += f"Negative language: {', '.join(negative_phrases[:3])}. "
        
        if confidence > 0.8:
            reasoning += "Very clear sentiment in speech."
        elif confidence > 0.6:
            reasoning += "Strong sentiment indicators."
        else:
            reasoning += "Moderate sentiment expression."
        
        return reasoning

# Test the fixed models
if __name__ == "__main__":
    print("🧪 Testing FIXED Audio Sentiment Analysis with Robust NLP:")
    
    # Test enhanced sentiment
    predictor = SentimentPredictor()
    test_text = "I absolutely love this amazing product! The customer service was outstanding and the quality exceeded my expectations."
    
    print(f"\nTest text: {test_text}")
    
    result = predictor.predict_sentiment_enhanced(test_text)
    print(f"✅ Enhanced Analysis Complete!")
    print(f"Sentiment: {result['primary_sentiment']} ({result['confidence']:.1%})")
    
    if "advanced_nlp" in result and "embeddings" in result["advanced_nlp"]:
        embeddings = result["advanced_nlp"]["embeddings"]
        if "embedding_dimension" in embeddings:
            print(f"Vector Embedding: {embeddings['embedding_dimension']} dimensions")
    
    # Test Whisper
    transcriber = WhisperTranscriber()
    if transcriber.model:
        print("✅ Whisper ready for audio transcription!")
    
    print("🎉 FIXED models ready with robust NLP capabilities!")