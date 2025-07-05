from flask import Flask, request, render_template, jsonify
import os
import sys

# Add the training directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))
from predict import process_video

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        if not video.filename.endswith('.mp4'):
            return jsonify({'error': 'Only .mp4 files are supported'}), 400

        video_path = os.path.join('uploads', video.filename)
        os.makedirs('uploads', exist_ok=True)
        video.save(video_path)

        try:
            results = process_video(video_path)
            os.remove(video_path)
            return jsonify(results)
        except Exception as e:
            os.remove(video_path)
            return jsonify({'error': f"Error processing video: {str(e)}"}), 500

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)