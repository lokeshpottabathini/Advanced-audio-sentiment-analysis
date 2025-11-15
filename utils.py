import librosa
import numpy as np
import os
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio file processing and analysis utilities"""
    
    def __init__(self):
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.webm', '.ogg', '.flac']
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file properties and quality
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio analysis results
        """
        try:
            logger.info(f"🔍 Analyzing audio file: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                return {
                    "analysis_status": "failed",
                    "error": f"File not found: {audio_path}"
                }
            
            # Get file info
            file_size = os.path.getsize(audio_path)
            filename = os.path.basename(audio_path)
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Load audio file using librosa
            try:
                # Load audio with librosa (handles most formats)
                audio_data, sample_rate = librosa.load(audio_path, sr=None)
                logger.info(f"✅ Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")
                
            except Exception as load_error:
                logger.error(f"❌ Failed to load audio: {load_error}")
                return {
                    "analysis_status": "failed",
                    "error": f"Could not load audio: {str(load_error)}"
                }
            
            # Calculate basic properties
            duration = len(audio_data) / sample_rate
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
            
            # Audio quality analysis
            quality_metrics = self._analyze_audio_quality(audio_data, sample_rate)
            
            # Compile results
            analysis_results = {
                "analysis_status": "success",
                "filename": filename,
                "format": file_extension,
                "file_size": file_size,
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "total_samples": len(audio_data),
                **quality_metrics
            }
            
            logger.info(f"✅ Audio analysis complete: {duration:.1f}s, {sample_rate}Hz, Quality: {quality_metrics.get('quality_score', 0):.2f}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Audio analysis failed: {e}")
            return {
                "analysis_status": "failed",
                "error": str(e)
            }
    
    def _analyze_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze audio quality metrics
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # RMS (Root Mean Square) - overall volume
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Peak amplitude
            peak_amplitude = np.max(np.abs(audio_data))
            
            # Dynamic range (difference between loudest and quietest parts)
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            
            # Signal-to-noise ratio estimate (simplified)
            # Calculate noise floor (bottom 10% of amplitudes)
            sorted_amplitudes = np.sort(np.abs(audio_data))
            noise_floor = np.mean(sorted_amplitudes[:int(len(sorted_amplitudes) * 0.1)])
            signal_power = rms
            snr_estimate = 20 * np.log10(signal_power / max(noise_floor, 1e-10))
            
            # Zero crossing rate (measure of noisiness)
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zero_crossing_rate = zero_crossings / len(audio_data)
            
            # Spectral analysis
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            avg_spectral_centroid = np.mean(spectral_centroid)
            
            # Overall quality score (0-1)
            # Factors: good SNR, reasonable peak levels, spectral content
            quality_factors = []
            
            # SNR factor (good if > 20dB)
            snr_factor = min(1.0, max(0.0, (snr_estimate - 10) / 30))
            quality_factors.append(snr_factor)
            
            # Peak level factor (good if not too quiet or clipped)
            if peak_amplitude > 0.95:  # Likely clipped
                peak_factor = 0.3
            elif peak_amplitude < 0.01:  # Too quiet
                peak_factor = 0.4
            else:
                peak_factor = 1.0
            quality_factors.append(peak_factor)
            
            # Sample rate factor (higher is generally better for speech)
            if sample_rate >= 44100:
                sr_factor = 1.0
            elif sample_rate >= 22050:
                sr_factor = 0.9
            elif sample_rate >= 16000:
                sr_factor = 0.8
            else:
                sr_factor = 0.6
            quality_factors.append(sr_factor)
            
            # Overall quality score
            quality_score = np.mean(quality_factors)
            
            return {
                "rms_amplitude": float(rms),
                "peak_amplitude": float(peak_amplitude),
                "dynamic_range": float(dynamic_range),
                "snr_estimate_db": float(snr_estimate),
                "zero_crossing_rate": float(zero_crossing_rate),
                "avg_spectral_centroid": float(avg_spectral_centroid),
                "quality_score": float(quality_score)
            }
            
        except Exception as e:
            logger.error(f"❌ Quality analysis failed: {e}")
            return {
                "quality_score": 0.5,  # Default neutral quality
                "error": str(e)
            }
    
    def is_supported_format(self, filename: str) -> bool:
        """
        Check if audio format is supported
        
        Args:
            filename: Name of audio file
            
        Returns:
            True if format is supported
        """
        file_extension = os.path.splitext(filename)[1].lower()
        return file_extension in self.supported_formats
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        Get basic audio file information without loading full audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Basic file information
        """
        try:
            if not os.path.exists(audio_path):
                return {"error": "File not found"}
            
            file_size = os.path.getsize(audio_path)
            filename = os.path.basename(audio_path)
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Try to get duration without loading full file
            try:
                duration = librosa.get_duration(path=audio_path)
            except:
                duration = 0.0
            
            return {
                "filename": filename,
                "format": file_extension,
                "file_size": file_size,
                "duration": duration,
                "supported": self.is_supported_format(filename)
            }
            
        except Exception as e:
            return {"error": str(e)}

class TextProcessor:
    """Text processing utilities for sentiment analysis"""
    
    def __init__(self):
        pass
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Processed text with metadata
        """
        try:
            logger.info("📝 Preprocessing text...")
            
            original_text = text
            original_length = len(text)
            
            # Basic cleaning
            # Remove extra whitespace
            cleaned_text = ' '.join(text.split())
            
            # Remove excessive punctuation (but keep some for sentiment)
            import re
            cleaned_text = re.sub(r'[!]{3,}', '!!!', cleaned_text)  # Max 3 exclamation marks
            cleaned_text = re.sub(r'[?]{3,}', '???', cleaned_text)  # Max 3 question marks
            cleaned_text = re.sub(r'[.]{4,}', '...', cleaned_text)  # Max 3 dots
            
            # Keep text as-is for sentiment analysis (minimal preprocessing)
            processed_text = cleaned_text
            
            # Calculate statistics
            words = processed_text.split()
            sentences = processed_text.count('.') + processed_text.count('!') + processed_text.count('?')
            
            result = {
                "original_text": original_text,
                "cleaned_text": processed_text,
                "original_length": original_length,
                "cleaned_length": len(processed_text),
                "word_count": len(words),
                "sentence_count": max(1, sentences),  # At least 1 sentence
                "preprocessing_applied": [
                    "whitespace_normalization",
                    "punctuation_normalization"
                ]
            }
            
            logger.info(f"✅ Text preprocessing complete: {original_length} → {len(processed_text)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Text preprocessing failed: {e}")
            return {
                "original_text": text,
                "cleaned_text": text,
                "original_length": len(text),
                "cleaned_length": len(text),
                "error": str(e)
            }
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text (simple implementation)
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        try:
            # Simple keyword extraction based on word frequency
            words = text.lower().split()
            
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
            }
            
            # Filter and count words
            word_freq = {}
            for word in words:
                # Remove punctuation
                clean_word = ''.join(c for c in word if c.isalnum())
                if len(clean_word) > 2 and clean_word not in stop_words:
                    word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]
            
            return keywords
            
        except Exception as e:
            logger.error(f"❌ Keyword extraction failed: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Audio Processing Utilities:\n")
    
    # Test AudioProcessor
    audio_processor = AudioProcessor()
    
    print("📁 Supported formats:", audio_processor.supported_formats)
    
    # Test TextProcessor
    text_processor = TextProcessor()
    
    test_text = "I absolutely love this amazing product!!! The quality is outstanding and the customer service was fantastic. I would definitely recommend this to everyone!"
    
    print(f"\nTest text: {test_text}")
    
    # Preprocess text
    processed = text_processor.preprocess_text(test_text)
    print(f"\nPreprocessed text: {processed['cleaned_text']}")
    print(f"Original length: {processed['original_length']}")
    print(f"Cleaned length: {processed['cleaned_length']}")
    print(f"Word count: {processed['word_count']}")
    print(f"Sentence count: {processed['sentence_count']}")
    
    # Extract keywords
    keywords = text_processor.extract_keywords(test_text)
    print(f"\nExtracted keywords: {', '.join(keywords)}")
    
    print("\n✅ Utility tests completed!")