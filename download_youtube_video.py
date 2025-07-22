```python
import argparse
import os
from pytube import YouTube
from pytube.exceptions import PytubeError

def download_youtube_video(url: str, output_path: str) -> None:
    """Download a YouTube video to the specified output path."""
    try:
        # Create YouTube object
        yt = YouTube(url)
        
        # Select the highest resolution MP4 stream
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            print("Error: No suitable MP4 stream found.")
            return
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Download the video
        print(f"Downloading: {yt.title}")
        stream.download(output_path=output_path)
        print(f"Download completed: {os.path.join(output_path, stream.default_filename)}")
        
    except PytubeError as e:
        print(f"Error downloading video: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download a YouTube video.")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--output", type=str, default="downloads", help="Output directory for the video (default: 'downloads')")
    args = parser.parse_args()

    # Download the video
    download_youtube_video(args.url, args.output)

if __name__ == "__main__":
    main()
