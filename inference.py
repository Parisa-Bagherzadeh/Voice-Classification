import numpy as np
import os
from argparse import ArgumentParser

from model import AudioClassifier



if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument("--model", type = str, default='model/weights.h5', help = "the path of .h5 file")
    parser.add_argument("--wav_file", type = str, default = "test_files/test.wav", help = "the path of wav file to inference")
    args = parser.parse_args()
    audioclassifier = AudioClassifier.predict(args.model, args.wav_file)
   