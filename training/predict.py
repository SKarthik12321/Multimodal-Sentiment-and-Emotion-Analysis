import torch
import cv2
import torchaudio
import numpy as np
from transformers import AutoTokenizer
from models import MultimodalSentimentModel
import os
from ffmpeg import input, output

# Load the trained model
model_path = "../models/model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first using train.py.")
model = MultimodalSentimentModel.load_model(model_path)
device = torch.device("cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def process_video(video_path, timeframe=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_chunk = int(fps * timeframe)

    results = []
    audio_path = video_path.replace('.mp4', '.wav')

    try:
        (
            input(video_path, v='0')
            .output(audio_path, acodec='pcm_s16le', ar='16000', ac='1')
            .run(quiet=True)
        )
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
        )
        mel_spec = mel_spectrogram(waveform)
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
        if mel_spec.size(2) < 300:
            padding = 300 - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        else:
            mel_spec = mel_spec[:, :, :300]

        audio_features = mel_spec.unsqueeze(0).to(device)
    except Exception as e:
        cap.release()
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise ValueError(f"Audio extraction failed: {str(e)}")

    emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    for start_frame in range(0, frame_count, frames_per_chunk):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        video_frames = []

        for _ in range(min(frames_per_chunk, frame_count - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            video_frames.append(frame)

        if not video_frames:
            break

        if len(video_frames) < 30:
            video_frames += [np.zeros_like(video_frames[0])] * (30 - len(video_frames))
        else:
            video_frames = video_frames[:30]

        video_frames = torch.FloatTensor(np.array(video_frames)).permute(0, 3, 1, 2).to(device)

        text = "Sample utterance"
        text_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            outputs = model(text_inputs, video_frames, audio_features)
            emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
            sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

            emotion_result = max(enumerate(emotion_probs), key=lambda x: x[1])
            sentiment_result = max(enumerate(sentiment_probs), key=lambda x: x[1])

            start_time = start_frame / fps
            end_time = min((start_frame + frames_per_chunk) / fps, frame_count / fps)
            results.append({
                'timeframe': f"{start_time:.2f}-{end_time:.2f} seconds",
                'emotion': emotion_map[emotion_result[0]],
                'emotion_confidence': emotion_result[1].item(),
                'sentiment': sentiment_map[sentiment_result[0]],
                'sentiment_confidence': sentiment_result[1].item()
            })

    cap.release()
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return results

if __name__ == "__main__":
    video_path = "../dataset/train/train_splits/dia1_utt1.mp4"
    if os.path.exists(video_path):
        results = process_video(video_path)
        print("\nVideo Analysis Results:")
        for result in results:
            print(result)
    else:
        print(f"Video file not found at: {video_path}. Please specify a valid video path.")