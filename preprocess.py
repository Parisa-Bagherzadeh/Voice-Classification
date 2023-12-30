from argparse import ArgumentParser

import os
import pydub

from model import AudioClassifier


def preprocss(raw_data_path, min_silence_len, silence_thresh, chunk_duration):

    path = os.listdir(raw_data_path)

    for file in path:
        voice = pydub.AudioSegment.from_file(os.path.join(raw_data_path, file))
        voice = voice.set_sample_width(2)
        voice = voice.set_frame_rate(48000)
        voice = voice.set_channels(1)
        chunks = pydub.silence.split_on_silence(voice, min_silence_len = min_silence_len, silence_thresh = silence_thresh)
        result = sum(chunks)
        file_name = file.split(".")[0]

        result.export("wav_data/" + file_name + ".wav", format = "wav")


    for file in os.listdir("wav_data"):
        audio = pydub.AudioSegment.from_file(os.path.join("wav_data", file))

        filename = file.split(".")[0]
        os.makedirs(os.path.join("dataset" , filename), exist_ok = True)
        chunks = pydub.utils.make_chunks(audio, chunk_duration)

        for i, chunk in enumerate(chunks):
            if len(chunk) >= chunk_duration:
                chunk.export(os.path.join("dataset", filename, f"{i}.wav"), format="wav")    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_data_path", type = str, default = "raw_data", help = "set the path of raw data")
    parser.add_argument("--min_silence_len", type = int, default = 2000, help = "enter min silence len in milisecond")
    parser.add_argument("--silence_thresh" , type = int, default = -45, help = "enter the silence_thresh")
    parser.add_argument("--chunk_duration", type = int, default = 1000, help = "enter the chunk duration in milisecond")

    args = parser.parse_args()
    audioclassifier = AudioClassifier()
    audioclassifier.preprocss(args.raw_data_path, args.min_silence_len, args.silence_thresh, args.chunk_duration)
    