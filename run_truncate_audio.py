import os
import argparse
from tqdm import tqdm
from pydub import AudioSegment


def main(args):
    audio_root_dir = args.audio_dir

    for score_name in tqdm(os.listdir(audio_root_dir)):
        score_dir = os.path.join(audio_root_dir, score_name)
        if not os.path.isdir(score_dir):
            continue

        for audio_name in os.listdir(score_dir):
            if not audio_name.endswith(".mp3"):
                continue

            audio_path = os.path.join(score_dir, audio_name)

            audio = AudioSegment.from_mp3(audio_path)
            audio = audio[:15 * 1000]
            audio.export(audio_path, format="mp3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='path to audio output directory')
    
    args = parser.parse_args()

    main(args)
