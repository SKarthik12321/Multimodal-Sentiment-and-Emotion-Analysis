import streamlit as st
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from transformers import AutoTokenizer
from inference import load_model
from video_processor import process_video
from report_generator import generate_pdf_report
import logging
import time
from datetime import datetime
import json
import base64
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Emotion and Sentiment emoji mappings
EMOTION_EMOJI = {
    "Happiness": "😊",
    "Sadness": "😢",
    "Anger": "😠",
    "Neutral": "😐",
    "Disgust": "🤢",
    "Fear": "😨",
    "Surprise": "😲"
}

SENTIMENT_EMOJI = {
    "Positive": "👍",
    "Negative": "👎",
    "Neutral": "😶"
}

# Set page config
st.set_page_config(page_title="Sentiment & Emotion Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(45deg, #FF4B4B, #FF9D9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 15px;
        padding: 10px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sub-title {
        font-size: 24px;
        color: #ddd;
        text-align: center;
        margin-bottom: 50px;
        font-style: italic;
    }
    .section-title {
        font-size: 28px;
        font-weight: bold;
        color: #eee;
        margin-top: 30px;
        margin-bottom: 20px;
        border-left: 5px solid #FF4B4B;
        padding-left: 12px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B, #FF7878);
        color: #fff;
        border-radius: 7px;
        padding: 14px 28px;
        border: none;
        box-shadow: 0 5px 8px rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #E04343, #FF6B6B);
        box-shadow: 0 7px 10px rgba(255,255,255,0.15);
        transform: translateY(-2px);
    }
    .card {
        background-color: #333333;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 5px 10px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        border: 1px solid #555;
        color: #fff;
    }
    .info-box {
        background-color: #333333;
        border-left: 4px solid #FF4B4B;
        padding: 18px;
        border-radius: 7px;
        margin-bottom: 25px;
        color: #ddd;
    }
    .metric-card {
        background: linear-gradient(135deg, #333, #444);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.5);
        color: #fff;
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 15px;
        color: #ddd;
    }
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #777;
        background-color: #444;
        color: #fff;
    }
    .stDataFrame table {
        background-color: #444;
        color: #fff;
    }
    [data-testid="stSidebar"] {
        background-color: #222;
        padding: 20px;
        color: #fff;
    }
    [data-testid="stSidebar"] h2 {
        color: #eee;
    }
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #ddd;
    }
    .stRadio > label > div > p {
        font-size: 16px;
        color: #ddd;
    }
    .stFileUploader input[type="file"] {
        color: #fff;
    }
    .stFileUploader label {
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", ["Analyzer", "About", "Help", "Settings", "Feedback"])

# Sidebar metadata
current_time = datetime.now().strftime("%Y-%m-%d")
project_title = "Sentiment & Emotion Analysis"
st.sidebar.markdown(f"**{project_title}**")
st.sidebar.markdown(f"**Date:** {current_time}")

# Main title and subtitle
st.markdown(f'<div class="main-title">{project_title}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unveil the Hidden Feelings in Your Videos</div>', unsafe_allow_html=True)

if page == "Analyzer":
    # Load model and tokenizer
    @st.cache_resource
    def load_resources():
        logger.info("Loading model and tokenizer...")
        try:
            model_path = "../models/model.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model = load_model(model_path)
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
            raise

    # Clear cache button
    if st.sidebar.button("Clear Cache", help="Reset the application and clear loaded models"):
        st.cache_resource.clear()
        st.sidebar.success("✅ Cache cleared! Please reload the app.")

    try:
        model, tokenizer = load_resources()
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {str(e)}")
        st.stop()

    # Video upload section
    st.markdown('<div class="section-title">Upload Your Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">For best results, upload videos with clear audio and good lighting. Supported formats: MP4, AVI, MOV.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        temp_video_path = "temp_video.mp4"
        try:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())
            logger.info(f"Saved uploaded video to {temp_video_path}")
        except Exception as e:
            st.error(f"Error saving video file: {str(e)}")
            st.stop()

        # Display video preview
        st.markdown('<div class="card">', unsafe_allow_html=True)
        try:
            st.video(temp_video_path)
        except Exception as e:
            st.warning(f"Could not display video preview: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Analysis settings
        st.markdown('<div class="section-title">Analysis Settings</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            interval = st.slider("Analysis interval (seconds)", min_value=1, max_value=10, value=5, step=1, help="Time interval between emotion analysis points")
        with col2:
            analysis_depth = st.select_slider("Analysis Depth", options=["Basic", "Standard", "Deep"], value="Standard", help="Controls the depth of emotional analysis")

        # Advanced settings
        with st.expander("Advanced Settings"):
            confidence_threshold = st.slider("Confidence Threshold (%)", min_value=0, max_value=100, value=50, help="Minimum confidence level to report an emotion")
            include_facial = st.checkbox("Include Facial Analysis", value=True, help="Analyze facial expressions in addition to speech")
            include_voice = st.checkbox("Include Voice Tone Analysis", value=True, help="Analyze voice tonality in addition to speech")
            save_analysis = st.checkbox("Save Analysis Results", value=False, help="Save the analysis results for future reference")
            show_confidence_breakdown = st.checkbox("Show Confidence Breakdown", value=False, help="Display detailed confidence scores for each emotion and sentiment")

        # Analyze button with progress indicator
        if st.button("Analyze Video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(101):
                progress_bar.progress(i)
                if i < 20:
                    status_text.text("Loading video frames...")
                elif i < 40:
                    status_text.text("Processing audio...")
                elif i < 60:
                    status_text.text("Extracting dialogues...")
                elif i < 80:
                    status_text.text("Analyzing emotions and sentiment...")
                else:
                    status_text.text("Finalizing results...")
                time.sleep(0.05)
            status_text.text("Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

            with st.spinner("Preparing results..."):
                try:
                    logger.info("Starting video analysis...")
                    timeline, duration = process_video(temp_video_path, model, tokenizer, interval=interval, confidence_threshold=confidence_threshold)
                    logger.info("Video analysis completed")

                    st.markdown("""
                        <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 30px; color: white;">
                            <h2 style="color: white; margin: 0;">✅ Analysis Complete!</h2>
                            <p style="margin: 10px 0 0 0;">Scroll down to see the detailed results</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Add emojis to timeline data
                    for entry in timeline:
                        entry['Emotion'] = f"{entry['emotion']} {EMOTION_EMOJI.get(entry['emotion'], '')}"
                        entry['Sentiment'] = f"{entry['sentiment']} {SENTIMENT_EMOJI.get(entry['sentiment'], '')}"

                    # Display timeline
                    st.markdown('<div class="section-title">Emotion and Sentiment Timeline</div>', unsafe_allow_html=True)
                    timeline_df = pd.DataFrame(timeline)
                    timeline_df['Time Range'] = timeline_df.apply(lambda row: f"{row['start_time']:.1f}s - {row['end_time']:.1f}s", axis=1)
                    display_df = timeline_df[['Time Range', 'Emotion', 'emotion_prob', 'Sentiment', 'sentiment_prob', 'dialogue']]
                    display_df = display_df.rename(columns={
                        'emotion_prob': 'Emotion Confidence',
                        'sentiment_prob': 'Sentiment Confidence',
                        'dialogue': 'Dialogue'
                    })
                    display_df['Emotion Confidence'] = display_df['Emotion Confidence'].apply(lambda x: f"{x:.2%}")
                    display_df['Sentiment Confidence'] = display_df['Sentiment Confidence'].apply(lambda x: f"{x:.2%}")
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.dataframe(display_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Confidence breakdown
                    if show_confidence_breakdown:
                        st.markdown('<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True)
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        for idx, entry in enumerate(timeline):
                            st.markdown(f"**Time Range: {entry['start_time']:.1f}s - {entry['end_time']:.1f}s**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Emotion Probabilities:**")
                                for emotion, prob in entry['emotion_probs'].items():
                                    st.markdown(f"- {emotion}: {prob:.2%}")
                            with col2:
                                st.markdown("**Sentiment Probabilities:**")
                                for sentiment, prob in entry['sentiment_probs'].items():
                                    st.markdown(f"- {sentiment}: {prob:.2%}")
                            st.markdown("---")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Key insights
                    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    dominant_emotion = timeline_df['Emotion'].value_counts().index[0]
                    avg_emotion_confidence = timeline_df['emotion_prob'].mean() * 100
                    dominant_sentiment = timeline_df['Sentiment'].value_counts().index[0]
                    emotion_changes = (timeline_df['Emotion'] != timeline_df['Emotion'].shift()).sum()
                    with col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Dominant Emotion</div>
                                <div class="metric-value">{dominant_emotion}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Avg. Confidence</div>
                                <div class="metric-value">{avg_emotion_confidence:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Dominant Sentiment</div>
                                <div class="metric-value">{dominant_sentiment}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Emotion Changes</div>
                                <div class="metric-value">{emotion_changes}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Interactive timeline visualization
                    st.markdown('<div class="section-title">Interactive Timeline Visualization</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    timeline_df['Emotion Label'] = timeline_df['Emotion'].apply(lambda x: x.split()[0])
                    timeline_df['Sentiment Label'] = timeline_df['Sentiment'].apply(lambda x: x.split()[0])
                    fig = px.line(timeline_df, x='start_time', y='Emotion Label', color='Emotion Label', title='Emotion Timeline',
                                  labels={'start_time': 'Time (seconds)', 'Emotion Label': 'Emotion'},
                                  color_discrete_map={"Happiness": "green", "Sadness": "blue", "Anger": "red", "Neutral": "grey", "Disgust": "purple", "Fear": "orange", "Surprise": "yellow"})
                    fig.update_layout(plot_bgcolor='#333333', paper_bgcolor='#333333', font_color='white')
                    st.plotly_chart(fig, use_container_width=True)
                    fig = px.line(timeline_df, x='start_time', y='Sentiment Label', color='Sentiment Label', title='Sentiment Timeline',
                                  labels={'start_time': 'Time (seconds)', 'Sentiment Label': 'Sentiment'},
                                  color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "grey"})
                    fig.update_layout(plot_bgcolor='#333333', paper_bgcolor='#333333', font_color='white')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Emotion transition heatmap
                    st.markdown('<div class="section-title">Emotion Transition Heatmap</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    emotions = timeline_df['Emotion Label'].unique()
                    transition_matrix = pd.DataFrame(0, index=emotions, columns=emotions)
                    for i in range(len(timeline_df) - 1):
                        current_emotion = timeline_df['Emotion Label'].iloc[i]
                        next_emotion = timeline_df['Emotion Label'].iloc[i + 1]
                        transition_matrix.loc[current_emotion, next_emotion] += 1
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Reds', ax=ax)
                    ax.set_title("Emotion Transition Heatmap", fontsize=16, pad=20, color='white')
                    ax.set_xlabel("To Emotion", fontsize=12, color='white')
                    ax.set_ylabel("From Emotion", fontsize=12, color='white')
                    ax.tick_params(colors='white')
                    fig.patch.set_facecolor('#333333')
                    ax.set_facecolor('#444')
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Summary report
                    st.markdown('<div class="section-title">Summary Report</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("Emotion Distribution")
                        emotion_counts = timeline_df['Emotion Label'].value_counts()
                        fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, title="Emotion Distribution",
                                     color=emotion_counts.index,
                                     color_discrete_map={"Happiness": "green", "Sadness": "blue", "Anger": "red", "Neutral": "grey", "Disgust": "purple", "Fear": "orange", "Surprise": "yellow"})
                        fig.update_layout(plot_bgcolor='#333333', paper_bgcolor='#333333', font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("Sentiment Distribution")
                        sentiment_counts = timeline_df['Sentiment Label'].value_counts()
                        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution",
                                     color=sentiment_counts.index,
                                     color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "grey"})
                        fig.update_layout(plot_bgcolor='#333333', paper_bgcolor='#333333', font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Summary text
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    dominant_emotion = emotion_counts.index[0]
                    dominant_emotion_pct = (emotion_counts[0] / sum(emotion_counts)) * 100
                    dominant_sentiment = sentiment_counts.index[0]
                    dominant_sentiment_pct = (sentiment_counts[0] / sum(sentiment_counts)) * 100
                    summary_text = f"""
                    ### Analysis Summary
                    The video predominantly shows **{dominant_emotion}** ({dominant_emotion_pct:.1f}%) with a **{dominant_sentiment}** sentiment ({dominant_sentiment_pct:.1f}%).
                    **Key Observations:**
                    - Total analysis duration: {duration:.2f} seconds
                    - Number of segments analyzed: {len(timeline)}
                    - Average emotion confidence: {avg_emotion_confidence:.1f}%
                    - Emotion changes detected: {emotion_changes}
                    **Recommendation:** This video is best suited for {dominant_sentiment.split()[0].lower()} content that evokes {dominant_emotion.lower()} responses.
                    **Additional Notes:** Results may vary based on video and audio quality.
                    """
                    st.markdown(summary_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Export results
                    st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        report_path = "sentiment_emotion_report.pdf"
                        generate_pdf_report(timeline, report_path)
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=f,
                                file_name="sentiment_emotion_report.pdf",
                                mime="application/pdf"
                            )
                    with col2:
                        csv = timeline_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📊 Export CSV Data",
                            data=csv,
                            file_name="sentiment_emotion_analysis.csv",
                            mime="text/csv"
                        )
                    with col3:
                        json_data = timeline_df.to_json(orient="records")
                        st.download_button(
                            label="🔄 Export JSON Data",
                            data=json_data,
                            file_name="sentiment_emotion_analysis.json",
                            mime="application/json"
                        )
                    with col4:
                        raw_data = pd.DataFrame(timeline)
                        raw_json = raw_data.to_json(orient="records")
                        st.download_button(
                            label="📦 Export Raw Data",
                            data=raw_json,
                            file_name="raw_analysis_data.json",
                            mime="application/json"
                        )

                    # Save analysis if selected
                    if save_analysis:
                        with open(f"analysis_{current_time}.json", "w") as f:
                            json.dump(timeline, f)
                        st.success("Analysis results saved successfully!")

                except Exception as e:
                    logger.error(f"Error during analysis: {e}")
                    st.error(f"Error during analysis: {str(e)}")

            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                logger.info(f"Removed temporary video file: {temp_video_path}")

elif page == "About":
    st.markdown('<div class="section-title">About Sentiment & Emotion Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### What is This Project?
    This project is designed to analyze emotions and sentiment in video content using AI. It processes videos to identify and track emotions expressed through facial expressions, voice tone, and spoken words, providing a comprehensive understanding of the video's emotional landscape.
    ### Key Features
    - **Comprehensive Emotion Detection**: Detects Happiness 😊, Sadness 😢, Anger 😠, Neutral 😐, and more.
    - **Sentiment Analysis**: Categorizes overall sentiment as Positive 👍, Negative 👎, or Neutral 😶.
    - **Multi-Modal Analysis**: Integrates facial expression, voice tone, and textual content analysis.
    - **Interactive Visualizations**: Explore emotional and sentiment trends with interactive charts.
    - **Customizable Settings**: Adjust analysis parameters for specific needs.
    - **Advanced Voice Analysis**: Identifies nuances in voice for deeper emotional insights.
    - **Dynamic Reporting**: Generate detailed PDF reports with visualizations.
    ### How It Works
    1. **Video Upload**: Upload a video file in MP4, AVI, or MOV format.
    2. **Analysis Configuration**: Set the analysis interval and other parameters.
    3. **Processing**: The system extracts frames, audio, and text.
    4. **Emotion and Sentiment Detection**: AI models analyze these elements.
    5. **Results Visualization**: View results in interactive timelines, charts, and summaries.
    ### Potential Applications
    - **Content Creation**: Enhance the emotional impact of videos.
    - **Marketing**: Test advertisement effectiveness.
    - **Education**: Analyze lecture emotional tone.
    - **Research**: Study human emotions in various contexts.
    - **Customer Service**: Analyze interactions for improvement.
    - **Therapy and Counseling**: Gain insights into emotional states.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Help":
    st.markdown('<div class="section-title">Getting Started</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### How to Use the Sentiment & Emotion Analyzer
    1. **Upload Your Video**: Select a video in MP4, AVI, or MOV format.
    2. **Configure Analysis Settings**:
        - **Analysis Interval**: Adjust the time interval between analysis points.
        - **Analysis Depth**: Choose between Basic, Standard, and Deep.
        - **Advanced Settings**:
            - **Confidence Threshold**: Set the minimum confidence level.
            - **Include Facial Analysis**: Enable/disable facial expression analysis.
            - **Include Voice Tone Analysis**: Enable/disable voice tone analysis.
    3. **Run Analysis**: Click "Analyze Video" to process the video.
    4. **Review Results**:
        - **Timeline**: See emotional and sentiment changes over time.
        - **Key Insights**: View dominant emotions, sentiments, and more.
        - **Visualizations**: Explore interactive charts.
        - **Summary Report**: Review distributions and summaries.
    5. **Export Results**: Download as PDF, CSV, JSON, or raw data.
    ### Tips for Best Results
    - Use videos with clear audio and good lighting.
    - Adjust the analysis interval for detailed tracking.
    - Experiment with confidence thresholds for accuracy.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Frequently Asked Questions</div>', unsafe_allow_html=True)
    with st.expander("What emotions and sentiments can this project detect?"):
        st.markdown("""
        This project detects the following emotions:
        - Happiness 😊
        - Sadness 😢
        - Anger 😠
        - Neutral 😐
        - Disgust 🤢
        - Fear 😨
        - Surprise 😲
        It also analyzes sentiment as:
        - Positive 👍
        - Negative 👎
        - Neutral 😶
        """)
    with st.expander("How accurate is the detection?"):
        st.markdown("""
        Accuracy depends on video, audio, and lighting quality. Confidence scores help assess reliability.
        """)
    with st.expander("Is my video data stored?"):
        st.markdown("""
        No, video data is processed temporarily and deleted afterward.
        """)
    with st.expander("Can I adjust the analysis settings?"):
        st.markdown("""
        Yes, you can adjust the interval, confidence threshold, and enable/disable facial and voice tone analysis.
        """)
    with st.expander("Is there a limit to the number of videos?"):
        st.markdown("""
        No limit, as processing occurs on your device.
        """)
    with st.expander("What factors affect video quality?"):
        st.markdown("""
        - Audio and lighting conditions.
        - Resolution clarity.
        """)

elif page == "Settings":
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Customize Your Experience")
    theme = st.selectbox("Theme", ["Dark", "Light"])
    if theme == "Light":
        st.markdown("""
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            .card {
                background-color: #f0f0f0;
                color: #000000;
            }
            </style>
        """, unsafe_allow_html=True)
    st.markdown("### Model Settings")
    st.slider("Model Confidence Adjustment", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Adjust the model's confidence threshold globally")
    st.markdown("### Export Preferences")
    st.checkbox("Include Raw Probabilities in Exports", value=True, help="Include raw probability scores in exported data")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Feedback":
    st.markdown('<div class="section-title">Provide Feedback</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### We Value Your Feedback!")
    user_name = st.text_input("Your Name (Optional)")
    user_email = st.text_input("Your Email (Optional)")
    feedback_rating = st.slider("Rate Your Experience (1-5)", min_value=1, max_value=5, value=3)
    feedback_comments = st.text_area("Your Feedback", help="Let us know how we can improve!")
    if st.button("Submit Feedback"):
        feedback_data = {
            "name": user_name,
            "email": user_email,
            "rating": feedback_rating,
            "comments": feedback_comments,
            "timestamp": current_time
        }
        with open(f"feedback_{current_time}_{user_name or 'anonymous'}.json", "w") as f:
            json.dump(feedback_data, f)
        st.success("Thank you for your feedback!")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")