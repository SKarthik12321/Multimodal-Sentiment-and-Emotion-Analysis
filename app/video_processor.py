import cv2
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from inference import predict_emotion_sentiment
    logger.info("Successfully imported predict_emotion_sentiment from inference")
except ImportError as e:
    logger.error(f"Failed to import predict_emotion_sentiment: {e}")
    raise

def extract_audio_from_video(video_path):
    logger.info(f"Extracting audio from video: {video_path}")
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    try:
        video = AudioSegment.from_file(video_path, format="mp4")
        video.export(temp_audio_path, format="wav")
        logger.info(f"Audio extracted to: {temp_audio_path}")
        return temp_audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None

def speech_to_text(audio_path, start_time, end_time):
    logger.info(f"Converting speech to text for time range {start_time}s to {end_time}s")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source, offset=start_time, duration=end_time - start_time)
            text = recognizer.recognize_google(audio)
            logger.info(f"Speech-to-text result: {text}")
            return text
    except (sr.UnknownValueError, sr.RequestError) as e:
        logger.warning(f"Speech-to-text error: {e}")
        return None

def process_video(video_path, model, tokenizer, interval=5, confidence_threshold=0.5):
    logger.info(f"Processing video: {video_path} with interval {interval}s")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file")
        raise ValueError("Could not open video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.info(f"Video duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
    audio_path = extract_audio_from_video(video_path)
    has_audio = audio_path is not None
    logger.info(f"Has audio: {has_audio}")
    timeline = []
    current_time = 0
    frame_idx = 0
    while current_time < duration:
        frame_idx = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at {current_time}s")
            break
        dialogue = None
        if has_audio:
            dialogue = speech_to_text(audio_path, current_time, min(current_time + interval, duration))
        text = dialogue if dialogue else " "
        logger.info(f"Predicting for time {current_time}s with text: {text}")
        try:
            prediction = predict_emotion_sentiment(model, frame, text, tokenizer)
            if prediction['emotion_prob'] < confidence_threshold / 100:
                prediction['emotion'] = 'Neutral'
                prediction['emotion_prob'] = 0.0
            if prediction['sentiment_prob'] < confidence_threshold / 100:
                prediction['sentiment'] = 'Neutral'
                prediction['sentiment_prob'] = 0.0
            logger.info(f"Prediction: {prediction}")
        except Exception as e:
            logger.error(f"Prediction error at {current_time}s: {e}")
            raise
        timeline.append({
            'start_time': current_time,
            'end_time': min(current_time + interval, duration),
            'emotion': prediction['emotion'],
            'emotion_prob': prediction['emotion_prob'],
            'sentiment': prediction['sentiment'],
            'sentiment_prob': prediction['sentiment_prob'],
            'dialogue': dialogue,
            'emotion_probs': prediction['emotion_probs'],
            'sentiment_probs': prediction['sentiment_probs']
        })
        current_time += interval
    cap.release()
    if has_audio and os.path.exists(audio_path):
        os.remove(audio_path)
        logger.info(f"Removed temporary audio file: {audio_path}")
    return timeline, duration