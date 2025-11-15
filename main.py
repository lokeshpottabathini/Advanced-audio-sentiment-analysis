import streamlit as st
import json
from datetime import datetime
import tempfile
import os
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Import our enhanced modules
from models import SentimentPredictor, WhisperTranscriber
from utils import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="Advanced Audio Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

def main():
    # Header with styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🧠 Advanced Audio Sentiment Analysis</h1>
        <p style="color: white; margin: 0; opacity: 0.9;">
            Audio → Transcription → Vector Embeddings → NLP Analysis → Similarity Search
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    render_sidebar()
    
    # Main interface
    render_main_interface()

def render_sidebar():
    """Render sidebar with advanced settings"""
    with st.sidebar:
        st.header("🎛️ Advanced System Status")
        
        # Model status
        st.subheader("🤖 AI Models")
        try:
            # Check models
            sentiment_predictor = SentimentPredictor()
            if hasattr(sentiment_predictor, 'models') and sentiment_predictor.models:
                if 'roberta' in sentiment_predictor.models:
                    st.success("✅ RoBERTa: Ready")
                if 'bert' in sentiment_predictor.models:
                    st.success("✅ BERT: Ready")
                if 'vader' in sentiment_predictor.models:
                    st.success("✅ VADER: Ready")
            
            # Check advanced NLP
            if hasattr(sentiment_predictor, 'nlp_analyzer'):
                st.success("✅ Advanced NLP: Ready")
                if hasattr(sentiment_predictor.nlp_analyzer, 'sentence_transformer') and sentiment_predictor.nlp_analyzer.sentence_transformer:
                    st.success("✅ Sentence Transformer: Ready")
                if hasattr(sentiment_predictor.nlp_analyzer, 'nlp_spacy') and sentiment_predictor.nlp_analyzer.nlp_spacy:
                    st.success("✅ spaCy: Ready")
                if hasattr(sentiment_predictor.nlp_analyzer, 'summarization_pipeline') and sentiment_predictor.nlp_analyzer.summarization_pipeline:
                    st.success("✅ BART Summarizer: Ready")
            
            # Check Whisper
            transcriber = WhisperTranscriber()
            if transcriber.model:
                st.success("✅ Whisper: Ready")
                
        except Exception as e:
            st.error(f"❌ Model Error: {str(e)[:50]}")
        
        # Analysis settings
        st.subheader("📊 Advanced Analysis Options")
        enable_embeddings = st.checkbox("Generate Vector Embeddings", value=True)
        enable_summarization = st.checkbox("Text Summarization", value=True)
        enable_nltk = st.checkbox("NLTK Analysis", value=True)
        enable_spacy = st.checkbox("spaCy Analysis", value=True)
        enable_similarity = st.checkbox("Similarity Search", value=True)
        show_embeddings_viz = st.checkbox("Visualize Embeddings", value=True)
        
        # Store settings
        st.session_state.enable_embeddings = enable_embeddings
        st.session_state.enable_summarization = enable_summarization
        st.session_state.enable_nltk = enable_nltk
        st.session_state.enable_spacy = enable_spacy
        st.session_state.enable_similarity = enable_similarity
        st.session_state.show_embeddings_viz = show_embeddings_viz
        
        # Advanced settings
        st.subheader("⚙️ Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
            index=0
        )
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        summary_length = st.selectbox("Summary Length", ["Short", "Medium", "Long"], index=1)
        
        st.session_state.embedding_model = embedding_model
        st.session_state.similarity_threshold = similarity_threshold
        st.session_state.summary_length = summary_length

def render_main_interface():
    """Render main interface with enhanced options"""
    
    # Input methods
    st.header("📥 Audio Input")
    
    input_method = st.radio(
        "Select input method:",
        ["📁 Upload Audio File", "✍️ Text Only (for testing NLP features)"],
        horizontal=True
    )
    
    audio_file = None
    text_input = None
    
    if input_method == "📁 Upload Audio File":
        audio_file = st.file_uploader(
            "Upload your audio file",
            type=['mp3', 'wav', 'm4a', 'webm', 'ogg', 'flac'],
            help="Supported formats: MP3, WAV, M4A, WebM, OGG, FLAC"
        )
        
        if audio_file is not None:
            # Display file info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 Filename", audio_file.name)
            with col2:
                file_size_mb = audio_file.size / (1024 * 1024)
                st.metric("📊 Size", f"{file_size_mb:.1f} MB")
            with col3:
                st.metric("🎵 Type", audio_file.type)
            
            # Audio player
            st.audio(audio_file, format=audio_file.type)
    
    else:  # Text input for testing NLP features
        text_input = st.text_area(
            "Enter text for advanced NLP analysis:",
            height=150,
            placeholder="Example: I'm absolutely thrilled with this new product! The quality is outstanding and it exceeded all my expectations. The customer service was amazing and the delivery was incredibly fast. I would highly recommend this to everyone looking for premium quality.",
            help="Enter longer text to see better NLP analysis results"
        )
        
        if text_input:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Characters", len(text_input))
            with col2:
                st.metric("🔤 Words", len(text_input.split()))
            with col3:
                sentences = text_input.count('.') + text_input.count('!') + text_input.count('?')
                st.metric("📄 Sentences", max(1, sentences))
    
    # Analysis button
    st.divider()
    
    if st.button("🚀 Start Advanced Analysis", type="primary", use_container_width=True):
        if audio_file is not None:
            run_advanced_audio_analysis(audio_file)
        elif text_input and text_input.strip():
            run_advanced_text_analysis(text_input.strip())
        else:
            st.error("❌ Please upload an audio file or enter text to analyze.")

def run_advanced_audio_analysis(audio_file):
    """Run complete advanced audio analysis"""
    
    st.header("🔄 Advanced Audio Analysis in Progress")
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    audio_path = None
    
    try:
        # Step 1: Save and process audio file
        progress_bar.progress(10)
        status_text.info("📁 Processing audio file...")
        
        file_extension = os.path.splitext(audio_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
        
        # Step 2: Audio analysis
        progress_bar.progress(20)
        status_text.info("🎵 Analyzing audio properties...")
        
        audio_processor = AudioProcessor()
        audio_analysis = audio_processor.analyze_audio(audio_path)
        
        # Step 3: Speech-to-text transcription
        progress_bar.progress(40)
        status_text.info("🗣️ Transcribing speech with Whisper...")
        
        transcriber = WhisperTranscriber()
        transcription_result = transcriber.transcribe(audio_path)
        
        if transcription_result.get('error'):
            st.error(f"❌ Transcription failed: {transcription_result['error']}")
            return
        
        transcript = transcription_result['text']
        
        # Step 4: Advanced sentiment analysis with NLP
        progress_bar.progress(60)
        status_text.info("🧠 Running advanced sentiment & NLP analysis...")
        
        sentiment_predictor = SentimentPredictor()
        enhanced_results = sentiment_predictor.predict_sentiment_enhanced(
            transcript, 
            include_nlp_analysis=True
        )
        
        # Step 5: Compile results
        progress_bar.progress(90)
        status_text.info("📊 Compiling advanced results...")
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "audio_info": {
                "filename": audio_file.name,
                "size_mb": audio_file.size / (1024 * 1024),
                "duration": audio_analysis.get('duration', 0),
                "quality": audio_analysis.get('quality_score', 0)
            },
            "transcription": transcription_result,
            "enhanced_sentiment": enhanced_results
        }
        
        # Step 6: Display results
        progress_bar.progress(100)
        status_text.success("🎉 Advanced analysis complete!")
        
        # Clear progress and show results
        progress_container.empty()
        display_advanced_results(final_results)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis error: {e}")
    
    finally:
        # Clean up
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

def run_advanced_text_analysis(text):
    """Run advanced text-only analysis"""
    
    st.header("🔄 Advanced Text Analysis in Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Text preprocessing
        progress_bar.progress(25)
        status_text.info("📝 Preprocessing text...")
        
        # Step 2: Advanced sentiment & NLP analysis
        progress_bar.progress(50)
        status_text.info("🧠 Running advanced sentiment & NLP analysis...")
        
        sentiment_predictor = SentimentPredictor()
        enhanced_results = sentiment_predictor.predict_sentiment_enhanced(
            text,
            include_nlp_analysis=True
        )
        
        # Step 3: Compile results
        progress_bar.progress(75)
        status_text.info("📊 Compiling results...")
        
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "audio_info": {"source": "text_input", "type": "direct_text"},
            "transcription": {
                "text": text,
                "confidence": 1.0,
                "language": "detected",
                "processing_time": 0.0
            },
            "enhanced_sentiment": enhanced_results
        }
        
        # Step 4: Display results
        progress_bar.progress(100)
        status_text.success("🎉 Advanced analysis complete!")
        
        # Show results
        display_advanced_results(final_results)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")

def display_advanced_results(results):
    """Display comprehensive advanced results"""
    
    st.header("🧠 Advanced Analysis Results")
    
    enhanced_sentiment = results.get("enhanced_sentiment", {})
    advanced_nlp = enhanced_sentiment.get("advanced_nlp", {})
    
    # Main sentiment metrics
    display_main_sentiment_metrics(enhanced_sentiment)
    
    # Audio/Text info
    if results.get("audio_info", {}).get("filename"):
        display_audio_info_section(results["audio_info"])
    
    # Transcription section
    display_transcription_section(results.get("transcription", {}))
    
    # Vector Embeddings Section
    if st.session_state.get('enable_embeddings', True) and advanced_nlp.get("embeddings"):
        display_embeddings_section(advanced_nlp["embeddings"], advanced_nlp.get("vector_stats", {}))
    
    # Summarization Section
    if st.session_state.get('enable_summarization', True) and advanced_nlp.get("summaries"):
        display_summarization_section(advanced_nlp["summaries"])
    
    # NLTK Analysis Section
    if st.session_state.get('enable_nltk', True) and advanced_nlp.get("nltk_analysis"):
        display_nltk_section(advanced_nlp["nltk_analysis"])
    
    # spaCy Analysis Section
    if st.session_state.get('enable_spacy', True) and advanced_nlp.get("spacy_analysis"):
        display_spacy_section(advanced_nlp["spacy_analysis"])
    
    # Similarity Search Section
    if st.session_state.get('enable_similarity', True) and advanced_nlp.get("similarity_analysis"):
        display_similarity_section(advanced_nlp["similarity_analysis"])
    
    # Semantic Features Section
    if advanced_nlp.get("semantic_features"):
        display_semantic_features_section(advanced_nlp["semantic_features"])
    
    # Export section
    display_advanced_export_section(results)
    
    # Success message
    st.success("🎉 **Advanced audio sentiment analysis completed successfully!** All NLP components working perfectly.")

def display_embeddings_section(embeddings_data, vector_stats):
    """Display vector embeddings analysis"""
    
    st.subheader("🧮 Vector Embeddings Analysis")
    
    if "error" in embeddings_data:
        st.error(f"❌ Embeddings error: {embeddings_data['error']}")
        return
    
    # Embedding statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Dimensions", embeddings_data.get("embedding_dimension", 0))
    
    with col2:
        st.metric("📄 Sentences", embeddings_data.get("num_sentences", 0))
    
    with col3:
        norm = vector_stats.get("text_embedding_norm", 0)
        st.metric("🔢 Vector Norm", f"{norm:.3f}")
    
    with col4:
        mean_val = vector_stats.get("text_embedding_mean", 0)
        st.metric("📊 Mean Value", f"{mean_val:.3f}")
    
    # Show embedding visualization if enabled
    if st.session_state.get('show_embeddings_viz', True):
        display_embedding_visualization(embeddings_data)
    
    # Sentence similarities
    if "sentence_similarities" in vector_stats:
        st.write("**📈 Sentence Similarity Matrix:**")
        similarity_matrix = np.array(vector_stats["sentence_similarities"]["similarity_matrix"])
        
        # Create heatmap
        fig = px.imshow(
            similarity_matrix,
            title="Sentence Similarity Heatmap (Cosine Similarity)",
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Similarity stats
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_sim = vector_stats["sentence_similarities"]["avg_similarity"]
            st.metric("📊 Avg Similarity", f"{avg_sim:.3f}")
        with col2:
            max_sim = vector_stats["sentence_similarities"]["max_similarity"]
            st.metric("📈 Max Similarity", f"{max_sim:.3f}")
        with col3:
            min_sim = vector_stats["sentence_similarities"]["min_similarity"]
            st.metric("📉 Min Similarity", f"{min_sim:.3f}")
    
    # Raw embedding display (first few dimensions)
    with st.expander("🔍 Raw Embedding Values (First 20 Dimensions)"):
        text_embedding = embeddings_data.get("text_embedding", [])
        if text_embedding:
            embedding_preview = text_embedding[:20]  # First 20 dimensions
            
            # Create bar chart of embedding values
            fig = go.Figure(data=go.Bar(
                x=[f"Dim_{i}" for i in range(len(embedding_preview))],
                y=embedding_preview,
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="Text Embedding Vector (First 20 Dimensions)",
                xaxis_title="Embedding Dimensions",
                yaxis_title="Values",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual values
            st.code(f"Full embedding vector shape: {len(text_embedding)} dimensions")

def display_embedding_visualization(embeddings_data):
    """Create embedding visualization"""
    
    sentences = embeddings_data.get("sentences", [])
    sentence_embeddings = embeddings_data.get("sentence_embeddings", [])
    
    if len(sentences) > 1 and sentence_embeddings:
        try:
            from sklearn.decomposition import PCA
            
            # Reduce dimensionality to 2D for visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(sentence_embeddings)
            
            # Create DataFrame for plotting
            df_viz = pd.DataFrame({
                'X': embeddings_2d[:, 0],
                'Y': embeddings_2d[:, 1],
                'Sentence': [s[:50] + "..." if len(s) > 50 else s for s in sentences],
                'Length': [len(s) for s in sentences]
            })
            
            # Create scatter plot
            fig = px.scatter(
                df_viz,
                x='X',
                y='Y',
                hover_data=['Sentence'],
                size='Length',
                title="Sentence Embeddings Visualization (PCA)",
                color='Length',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA explanation
            explained_var = pca.explained_variance_ratio_
            st.write(f"**PCA Explained Variance:** {explained_var[0]:.2%} (PC1), {explained_var[1]:.2%} (PC2)")
            
        except Exception as e:
            st.warning(f"⚠️ Visualization failed: {str(e)}")

def display_summarization_section(summaries):
    """Display text summarization results"""
    
    st.subheader("📝 Text Summarization")
    
    # Extractive summary
    if "extractive" in summaries:
        extractive = summaries["extractive"]
        
        st.write("**🔍 Extractive Summary (TextRank Algorithm):**")
        
        if "error" not in extractive:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(extractive.get("summary", "No summary available"))
            
            with col2:
                st.metric("📊 Compression Ratio", f"{extractive.get('compression_ratio', 0):.2%}")
                st.metric("📄 Sentences Used", f"{extractive.get('num_sentences_summary', 0)}/{extractive.get('num_sentences_original', 0)}")
            
            # Show method details
            st.write(f"**Method:** {extractive.get('method', 'TextRank')}")
        else:
            st.error(f"❌ Extractive summary error: {extractive['error']}")
    
    # Abstractive summary
    if "abstractive" in summaries:
        abstractive = summaries["abstractive"]
        
        st.write("**🤖 Abstractive Summary (BART Model):**")
        
        if "error" not in abstractive:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.success(abstractive.get("summary", "No summary available"))
            
            with col2:
                st.metric("📥 Input Length", abstractive.get("input_length", 0))
                st.metric("📤 Summary Length", abstractive.get("summary_length", 0))
            
            # Show model details
            st.write(f"**Model:** {abstractive.get('method', 'BART')}")
        else:
            st.warning(f"⚠️ Abstractive summary: {abstractive['error']}")

def display_nltk_section(nltk_analysis):
    """Display NLTK analysis results"""
    
    st.subheader("🔍 NLTK Analysis Results")
    
    if "error" in nltk_analysis:
        st.error(f"❌ NLTK error: {nltk_analysis['error']}")
        return
    
    # Tokenization stats
    tokenization = nltk_analysis.get("tokenization", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Sentences", tokenization.get("sentence_count", 0))
    with col2:
        st.metric("🔤 Words", tokenization.get("word_count", 0))
    with col3:
        st.metric("🔠 Filtered Words", tokenization.get("filtered_word_count", 0))
    with col4:
        avg_words = tokenization.get("avg_words_per_sentence", 0)
        st.metric("📊 Avg Words/Sentence", f"{avg_words:.1f}")
    
    # POS Analysis
    pos_analysis = nltk_analysis.get("pos_analysis", {})
    if pos_analysis:
        st.write("**📝 Part-of-Speech Analysis:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏷️ Nouns", pos_analysis.get("noun_count", 0))
        with col2:
            st.metric("⚡ Verbs", pos_analysis.get("verb_count", 0))
        with col3:
            st.metric("🎨 Adjectives", pos_analysis.get("adjective_count", 0))
        
        # POS distribution chart
        pos_dist = pos_analysis.get("pos_distribution", {})
        if pos_dist:
            fig = px.bar(
                x=list(pos_dist.keys()),
                y=list(pos_dist.values()),
                title="Part-of-Speech Distribution",
                labels={'x': 'POS Tags', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Named Entities
    named_entities = nltk_analysis.get("named_entities", [])
    if named_entities:
        st.write("**🏷️ Named Entities Detected:**")
        entities_df = pd.DataFrame(named_entities, columns=["Entity", "Type"])
        st.dataframe(entities_df, use_container_width=True)
    
    # Word frequency
    word_freq = nltk_analysis.get("word_frequency", {})
    if word_freq:
        st.write("**📊 Most Frequent Words:**")
        
        # Create word frequency chart
        words = list(word_freq.keys())[:10]  # Top 10
        frequencies = list(word_freq.values())[:10]
        
        fig = px.bar(
            x=frequencies,
            y=words,
            orientation='h',
            title="Top 10 Most Frequent Words",
            labels={'x': 'Frequency', 'y': 'Words'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # NLTK Sentiment
    nltk_sentiment = nltk_analysis.get("nltk_sentiment", {})
    if nltk_sentiment:
        st.write("**🎭 NLTK VADER Sentiment:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("😊 Positive", f"{nltk_sentiment.get('pos', 0):.3f}")
        with col2:
            st.metric("😐 Neutral", f"{nltk_sentiment.get('neu', 0):.3f}")
        with col3:
            st.metric("😔 Negative", f"{nltk_sentiment.get('neg', 0):.3f}")
        with col4:
            st.metric("🎯 Compound", f"{nltk_sentiment.get('compound', 0):.3f}")

def display_spacy_section(spacy_analysis):
    """Display spaCy analysis results"""
    
    st.subheader("🕷️ spaCy Analysis Results")
    
    if "error" in spacy_analysis:
        st.error(f"❌ spaCy error: {spacy_analysis['error']}")
        return
    
    # Token analysis
    token_analysis = spacy_analysis.get("token_analysis", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Tokens", token_analysis.get("total_tokens", 0))
    with col2:
        st.metric("🔤 Alpha Tokens", token_analysis.get("alpha_tokens", 0))
    with col3:
        st.metric("🛑 Stop Words", token_analysis.get("stop_words", 0))
    with col4:
        st.metric("📄 Sentences", spacy_analysis.get("sentence_count", 0))
    
    # Named Entities
    entities = spacy_analysis.get("entities", [])
    if entities:
        st.write("**🏷️ Named Entities (spaCy):**")
        
        # Create entities dataframe
        entities_df = pd.DataFrame(entities, columns=["Text", "Label", "Start", "End"])
        st.dataframe(entities_df, use_container_width=True)
        
        # Entity types distribution
        entity_labels = spacy_analysis.get("entity_labels", [])
        if entity_labels:
            label_counts = pd.Series(entity_labels).value_counts()
            
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Entity Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Noun Phrases
    noun_phrases = spacy_analysis.get("noun_phrases", [])
    if noun_phrases:
        st.write("**📝 Key Noun Phrases:**")
        phrases_str = ", ".join(noun_phrases[:15])  # Top 15
        st.info(phrases_str)
    
    # POS Distribution
    pos_distribution = spacy_analysis.get("pos_distribution", {})
    if pos_distribution:
        st.write("**📊 Part-of-Speech Distribution (spaCy):**")
        
        fig = px.bar(
            x=list(pos_distribution.keys()),
            y=list(pos_distribution.values()),
            title="spaCy POS Distribution",
            labels={'x': 'POS Tags', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_similarity_section(similarity_analysis):
    """Display similarity search capabilities"""
    
    st.subheader("🔗 Similarity Search Analysis")
    
    if "error" in similarity_analysis:
        st.error(f"❌ Similarity analysis error: {similarity_analysis['error']}")
        return
    
    # Search methods explanation
    st.write("**🔍 Similarity Search Methods Used:**")
    search_methods = similarity_analysis.get("similarity_methods_used", [])
    for method in search_methods:
        st.write(f"- {method}")
    
    # Embedding-based search results
    embedding_search = similarity_analysis.get("embedding_based_search", {})
    if embedding_search:
        st.write("**🧠 Semantic Search Results:**")
        
        query_sentence = embedding_search.get("query_sentence", "")
        if query_sentence:
            st.info(f"**Query Sentence:** {query_sentence}")
        
        similar_sentences = embedding_search.get("similar_sentences", [])
        if similar_sentences:
            # Display similar sentences
            for i, result in enumerate(similar_sentences, 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{i}.** {result.get('sentence', '')}")
                with col2:
                    similarity = result.get('similarity_score', 0)
                    st.metric("Similarity", f"{similarity:.3f}")
        
        search_method = embedding_search.get("search_method", "")
        st.caption(f"Method: {search_method}")
    
    # TF-IDF comparison
    tfidf_comparison = similarity_analysis.get("tfidf_comparison", {})
    if tfidf_comparison:
        st.write("**📊 TF-IDF Similarity Comparison:**")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_tfidf = tfidf_comparison.get("avg_tfidf_similarity", 0)
            st.metric("📊 Average TF-IDF Similarity", f"{avg_tfidf:.3f}")
        with col2:
            max_tfidf = tfidf_comparison.get("max_tfidf_similarity", 0)
            st.metric("📈 Max TF-IDF Similarity", f"{max_tfidf:.3f}")
    
    # Search capabilities explanation
    search_capabilities = similarity_analysis.get("search_capabilities", {})
    if search_capabilities:
        st.write("**🚀 Search Capabilities Demonstrated:**")
        
        with st.expander("🔍 Semantic Search Details"):
            semantic = search_capabilities.get("semantic_search", "")
            st.write(f"**Semantic Search:** {semantic}")
            
            keyword = search_capabilities.get("keyword_search", "")
            st.write(f"**Keyword Search:** {keyword}")
            
            hybrid = search_capabilities.get("hybrid_approach", "")
            st.write(f"**Hybrid Approach:** {hybrid}")

def display_semantic_features_section(semantic_features):
    """Display semantic features analysis"""
    
    st.subheader("🧠 Semantic Features Analysis")
    
    if "error" in semantic_features:
        st.error(f"❌ Semantic features error: {semantic_features['error']}")
        return
    
    # Readability metrics
    readability = semantic_features.get("readability", {})
    if readability:
        st.write("**📖 Readability Analysis:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_sent_len = readability.get("avg_sentence_length", 0)
            st.metric("📝 Avg Sentence Length", f"{avg_sent_len:.1f}")
        with col2:
            lexical_div = readability.get("lexical_diversity", 0)
            st.metric("📚 Lexical Diversity", f"{lexical_div:.3f}")
        with col3:
            avg_word_len = readability.get("avg_word_length", 0)
            st.metric("🔤 Avg Word Length", f"{avg_word_len:.1f}")
        with col4:
            semantic_coh = semantic_features.get("semantic_coherence", 0)
            st.metric("🧠 Semantic Coherence", f"{semantic_coh:.3f}")
    
    # Text complexity
    complexity = semantic_features.get("text_complexity", {})
    if complexity:
        st.write("**📊 Text Complexity Metrics:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📏 Long Sentences", complexity.get("long_sentences", 0))
        with col2:
            st.metric("🔤 Complex Words", complexity.get("complex_words", 0))
        with col3:
            unique_ratio = complexity.get("unique_word_ratio", 0)
            st.metric("🎯 Unique Word Ratio", f"{unique_ratio:.3f}")

def display_main_sentiment_metrics(enhanced_sentiment):
    """Display main sentiment metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    sentiment = enhanced_sentiment.get("primary_sentiment", "Unknown")
    confidence = enhanced_sentiment.get("confidence", 0.0)
    processing_time = enhanced_sentiment.get("processing_time", 0.0)
    enhancement_time = enhanced_sentiment.get("enhancement_time", 0.0)
    
    # Sentiment emoji
    emoji_map = {"Positive": "😊", "Negative": "😔", "Neutral": "😐", "Unknown": "❓"}
    sentiment_emoji = emoji_map.get(sentiment, "❓")
    
    with col1:
        st.metric("🎭 Sentiment", f"{sentiment_emoji} {sentiment}")
    
    with col2:
        st.metric("🎯 Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("⚡ Base Analysis", f"{processing_time:.2f}s")
    
    with col4:
        total_time = processing_time + enhancement_time
        st.metric("🧠 Total Analysis", f"{total_time:.2f}s")

def display_transcription_section(transcription):
    """Display transcription results"""
    
    st.subheader("📝 Speech Transcription Results")
    
    text = transcription.get('text', '')
    confidence = transcription.get('confidence', 0)
    language = transcription.get('language', 'unknown')
    
    if text:
        # Confidence and language
        col1, col2 = st.columns(2)
        
        with col1:
            if confidence > 0.8:
                st.success(f"✅ **Confidence**: {confidence:.1%}")
            elif confidence > 0.6:
                st.warning(f"⚠️ **Confidence**: {confidence:.1%}")
            else:
                st.error(f"❌ **Confidence**: {confidence:.1%}")
        
        with col2:
            st.info(f"🌍 **Language**: {language.title()}")
        
        # Full transcript
        st.write("**Complete Transcript:**")
        st.text_area(
            "transcript_display",
            value=text,
            height=120,
            disabled=True,
            label_visibility="collapsed"
        )
    else:
        st.warning("⚠️ No transcript available")

def display_audio_info_section(audio_info):
    """Display audio information"""
    
    st.subheader("🎵 Audio File Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**File Details:**")
        st.write(f"📄 **Name**: {audio_info.get('filename', 'Unknown')}")
        st.write(f"📊 **Size**: {audio_info.get('size_mb', 0):.1f} MB")
    
    with col2:
        st.write("**Properties:**")
        st.write(f"⏱️ **Duration**: {audio_info.get('duration', 0):.1f}s")
        st.write(f"✨ **Quality**: {audio_info.get('quality', 0):.2f}")
    
    with col3:
        st.write("**Status:**")
        st.success("✅ Successfully processed")
        st.success("✅ Analysis completed")

def display_advanced_export_section(results):
    """Display advanced export options"""
    
    st.subheader("📥 Export Advanced Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download Complete JSON", use_container_width=True):
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="💾 Save Complete Analysis",
                data=json_data,
                file_name=f"advanced_audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download NLP Summary", use_container_width=True):
            summary_text = create_advanced_summary_report(results)
            st.download_button(
                label="💾 Save NLP Report",
                data=summary_text,
                file_name=f"nlp_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("📈 Download Embeddings CSV", use_container_width=True):
            embeddings_csv = create_embeddings_csv(results)
            st.download_button(
                label="💾 Save Embeddings Data",
                data=embeddings_csv,
                file_name=f"embeddings_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def create_advanced_summary_report(results):
    """Create comprehensive advanced summary"""
    
    summary = "ADVANCED AUDIO SENTIMENT ANALYSIS REPORT\n"
    summary += "=" * 60 + "\n\n"
    
    summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += f"Analysis Type: Advanced Audio-to-Sentiment with NLP\n\n"
    
    # Audio information
    audio_info = results.get("audio_info", {})
    if audio_info.get('filename'):
        summary += f"AUDIO FILE INFORMATION:\n{'-' * 30}\n"
        summary += f"Filename: {audio_info.get('filename', 'Unknown')}\n"
        summary += f"Duration: {audio_info.get('duration', 0):.1f} seconds\n"
        summary += f"Quality Score: {audio_info.get('quality', 0):.2f}\n\n"
    
    # Transcription
    transcription = results.get("transcription", {})
    if transcription.get('text'):
        summary += f"SPEECH TRANSCRIPTION:\n{'-' * 25}\n"
        summary += f"Text: {transcription['text']}\n"
        summary += f"Confidence: {transcription.get('confidence', 0):.1%}\n"
        summary += f"Language: {transcription.get('language', 'Unknown')}\n\n"
    
    # Enhanced sentiment
    enhanced = results.get("enhanced_sentiment", {})
    if enhanced:
        summary += f"SENTIMENT ANALYSIS:\n{'-' * 20}\n"
        summary += f"Primary Sentiment: {enhanced.get('primary_sentiment', 'Unknown')}\n"
        summary += f"Confidence: {enhanced.get('confidence', 0):.1%}\n"
        summary += f"Ensemble Score: {enhanced.get('ensemble_score', 0):.3f}\n\n"
    
    # Advanced NLP
    advanced_nlp = enhanced.get("advanced_nlp", {})
    if advanced_nlp:
        summary += f"ADVANCED NLP ANALYSIS:\n{'-' * 25}\n"
        
        # Embeddings
        embeddings = advanced_nlp.get("embeddings", {})
        if embeddings and "embedding_dimension" in embeddings:
            summary += f"Vector Embeddings: {embeddings['embedding_dimension']} dimensions\n"
            summary += f"Sentences Analyzed: {embeddings.get('num_sentences', 0)}\n"
        
        # Summaries
        summaries = advanced_nlp.get("summaries", {})
        if summaries:
            if "extractive" in summaries and "summary" in summaries["extractive"]:
                summary += f"Extractive Summary: {summaries['extractive']['summary'][:200]}...\n"
            if "abstractive" in summaries and "summary" in summaries["abstractive"]:
                summary += f"Abstractive Summary: {summaries['abstractive']['summary'][:200]}...\n"
        
        # NLTK analysis
        nltk_analysis = advanced_nlp.get("nltk_analysis", {})
        if nltk_analysis and "tokenization" in nltk_analysis:
            tokenization = nltk_analysis["tokenization"]
            summary += f"Word Count: {tokenization.get('word_count', 0)}\n"
            summary += f"Sentence Count: {tokenization.get('sentence_count', 0)}\n"
        
        summary += "\n"
    
    summary += "=" * 60 + "\n"
    summary += "Report generated by Advanced Audio Sentiment Analysis System\n"
    
    return summary

def create_embeddings_csv(results):
    """Create CSV data for embeddings"""
    
    enhanced = results.get("enhanced_sentiment", {})
    advanced_nlp = enhanced.get("advanced_nlp", {})
    embeddings = advanced_nlp.get("embeddings", {})
    
    csv_lines = ["Dimension,Value,Type"]
    
    # Text embedding
    text_embedding = embeddings.get("text_embedding", [])
    for i, value in enumerate(text_embedding):
        csv_lines.append(f"Text_Dim_{i},{value:.6f},Text_Embedding")
    
    # Sentence embeddings (first sentence if available)
    sentence_embeddings = embeddings.get("sentence_embeddings", [])
    if sentence_embeddings:
        first_sentence_embedding = sentence_embeddings[0]
        for i, value in enumerate(first_sentence_embedding):
            csv_lines.append(f"Sent1_Dim_{i},{value:.6f},Sentence_Embedding")
    
    # Vector statistics
    vector_stats = advanced_nlp.get("vector_stats", {})
    for key, value in vector_stats.items():
        if isinstance(value, (int, float)):
            csv_lines.append(f"{key},{value:.6f},Vector_Stat")
    
    return '\n'.join(csv_lines)

if __name__ == "__main__":
    main()