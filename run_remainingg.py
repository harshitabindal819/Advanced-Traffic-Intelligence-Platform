import subprocess
import os

# Convert the already-processed video
input_video = r"runs\detect\predict\sample_video.avi"
output_video = "processed_sample_video.mp4"

if os.path.exists(input_video):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "panic",
        "-i", input_video,
        output_video
    ])
    print("✅ Video converted successfully!")
    print(f"📁 Saved as: {os.path.abspath(output_video)}")
else:
    # List what's in runs/detect to find the actual output
    for root, dirs, files in os.walk("runs"):
        for f in files:
            print(os.path.join(root, f))