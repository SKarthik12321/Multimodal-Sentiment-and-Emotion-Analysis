from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import io
import base64

def generate_pie_chart(data, labels, title, filename):
    plt.figure(figsize=(6, 6))
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_pdf_report(timeline, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Sentiment & Emotion Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, "Detailed Analysis of Video Emotions and Sentiments")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 100, "Emotion and Sentiment Timeline")
    y_position = height - 120
    c.setFont("Helvetica", 10)
    for entry in timeline:
        time_str = f"{entry['start_time']:.1f}s - {entry['end_time']:.1f}s"
        emotion_str = f"Emotion: {entry['emotion']} ({entry['emotion_prob']:.2%})"
        sentiment_str = f"Sentiment: {entry['sentiment']} ({entry['sentiment_prob']:.2%})"
        dialogue_str = f"Dialogue: {entry['dialogue']}" if entry['dialogue'] else "Dialogue: (Not available)"
        c.drawString(50, y_position, time_str)
        c.drawString(150, y_position, emotion_str)
        c.drawString(300, y_position, sentiment_str)
        c.drawString(50, y_position - 15, dialogue_str)
        y_position -= 30
        if y_position < 50:
            c.showPage()
            y_position = height - 50
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Summary Report")
    emotion_counts = pd.Series([entry['emotion'] for entry in timeline]).value_counts()
    emotion_labels = emotion_counts.index.tolist()
    emotion_data = emotion_counts.values.tolist()
    generate_pie_chart(emotion_data, emotion_labels, "Emotion Distribution", "emotion_pie.png")
    c.drawImage("emotion_pie.png", 50, height - 250, width=200, height=200)
    sentiment_counts = pd.Series([entry['sentiment'] for entry in timeline]).value_counts()
    sentiment_labels = sentiment_counts.index.tolist()
    sentiment_data = sentiment_counts.values.tolist()
    generate_pie_chart(sentiment_data, sentiment_labels, "Sentiment Distribution", "sentiment_pie.png")
    c.drawImage("sentiment_pie.png", 300, height - 250, width=200, height=200)
    dominant_emotion = emotion_counts.index[0]
    dominant_emotion_pct = (emotion_counts[0] / sum(emotion_counts)) * 100
    dominant_sentiment = sentiment_counts.index[0]
    dominant_sentiment_pct = (sentiment_counts[0] / sum(sentiment_counts)) * 100
    summary_text = (
        f"The video predominantly shows {dominant_emotion} ({dominant_emotion_pct:.1f}%) "
        f"with a {dominant_sentiment} sentiment ({dominant_sentiment_pct:.1f}%)."
    )
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 300, summary_text)
    c.save()
    if os.path.exists("emotion_pie.png"):
        os.remove("emotion_pie.png")
    if os.path.exists("sentiment_pie.png"):
        os.remove("sentiment_pie.png")