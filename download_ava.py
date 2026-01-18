import os
import csv
import subprocess
import sys

try:
    import yt_dlp
except ImportError:
    print("yt_dlp not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

class QuietLogger:
    def debug(self, msg):
        pass
    def info(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        print(msg)


def download_ava_audio(csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                video_ids.add(row[0])
    
    print(f"Found {len(video_ids)} unique videos in {csv_path}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1'
        ],
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
        'logger': QuietLogger(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        missing_ids = []
        valid_ids = []
        for vid in video_ids:
            file_path = os.path.join(output_dir, f"{vid}.wav")
            if os.path.exists(file_path):
                valid_ids.append(vid)
            else:
                missing_ids.append(vid)
        
        total_videos = len(video_ids)
        print(f"Need to download {len(missing_ids)} videos.")
        
        failed_ids = []
        already_have = len(valid_ids)
        success_count = already_have
        processed_missing = 0
        if missing_ids:
            total_missing = len(missing_ids)
            if tqdm is not None:
                iterator = tqdm(missing_ids, desc="Downloading", unit="video")
            else:
                iterator = missing_ids
            for idx, vid in enumerate(iterator, 1):
                processed_missing += 1
                file_path = os.path.join(output_dir, f"{vid}.wav")
                url = f"https://www.youtube.com/watch?v={vid}"
                status = "OK"
                try:
                    ydl.download([url])
                    if not os.path.exists(file_path):
                        failed_ids.append(vid)
                        status = "FAIL"
                    else:
                        success_count += 1
                        valid_ids.append(vid)
                except Exception:
                    failed_ids.append(vid)
                    status = "FAIL"
                current_processed = already_have + processed_missing
                valid_ids_str = ",".join(valid_ids)
                if tqdm is not None:
                    tqdm.write(
                        f"[{idx}/{total_missing}] {vid} {status}, "
                        f"valid_so_far={success_count}/{current_processed}, "
                        f"valid_ids=[{valid_ids_str}]"
                    )
                else:
                    print(
                        f"[{idx}/{total_missing}] {vid} {status}, "
                        f"valid_so_far={success_count}/{current_processed}, "
                        f"valid_ids=[{valid_ids_str}]"
                    )

        success_from_missing = len(missing_ids) - len(failed_ids)
        total_success = already_have + success_from_missing
        total_failed = total_videos - total_success
        if total_videos > 0:
            fail_ratio = total_failed / total_videos * 100.0
        else:
            fail_ratio = 0.0

        print(f"Download summary: success {total_success}/{total_videos}, "
              f"failed {total_failed} ({fail_ratio:.1f}%)")

    print("Download finished.")


if __name__ == "__main__":
    download_ava_audio("ava_speech_labels_v1.csv", "AVA_Audio")
