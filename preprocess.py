from argparse import ArgumentParser

import os
import pydub

from model import AudioClassifier



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_data_path", type = str, default = "raw_data", help = "set the path of raw data")
    parser.add_argument("--min_silence_len", type = int, default = 2000, help = "enter min silence len in milisecond")
    parser.add_argument("--silence_thresh" , type = int, default = -45, help = "enter the silence_thresh")
    parser.add_argument("--chunk_duration", type = int, default = 1000, help = "enter the chunk duration in milisecond")

    args = parser.parse_args()
    audioclassifier = AudioClassifier()
    audioclassifier.preprocss(args.raw_data_path, args.min_silence_len, args.silence_thresh, args.chunk_duration)
    