import numpy as np
import os
from argparse import ArgumentParser

import librosa

from tensorflow.keras.models import load_model



def test(model, wav_file):

    model = load_model(model)
    waveform, _ = librosa.load(wav_file, sr=None)

    desired_length = 48000
    resized_waveform = librosa.util.fix_length(waveform, size = desired_length)

    input_data = np.expand_dims(resized_waveform, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)


    prediction = model.predict(input_data)
    label = np.argmax(prediction)

    labels = os.listdir('dataset_voice')
    pred = labels[label]

    print(pred + "😍")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type = str, default='model/weights.h5', help = "the path of .h5 file")
    parser.add_argument("--wav_file", type = str, default = "test_files/test.wav", help = "the path of wav file to inference")
    args = parser.parse_args()

    test(args.model, args.wav_file)