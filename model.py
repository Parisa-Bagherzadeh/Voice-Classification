import os
import pydub
import numpy as np
import matplotlib.pyplot as plt
import librosa

from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

class AudioClassifier():

    def __init__(self):
        self.dataset_path = 'dataset'
        

    def preprocss(self, raw_data_path, min_silence_len, silence_thresh, chunk_duration):

        self.raw_data_path = raw_data_path
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.chunk_duration = chunk_duration
        path = os.listdir(self.raw_data_path)

        for file in path:
            voice = pydub.AudioSegment.from_file(os.path.join(self.raw_data_path, file))
            voice = voice.set_sample_width(2)
            voice = voice.set_frame_rate(48000)
            voice = voice.set_channels(1)
            chunks = pydub.silence.split_on_silence(voice, min_silence_len = self.min_silence_len, silence_thresh = self.silence_thresh)
            result = sum(chunks)
            file_name = file.split(".")[0]

            result.export("wav_data/" + file_name + ".wav", format = "wav")



        for file in os.listdir("wav_data"):
            audio = pydub.AudioSegment.from_file(os.path.join("wav_data", file))

            filename = file.split(".")[0]
            os.makedirs(os.path.join(self.dataset_path , filename), exist_ok = True)
            chunks = pydub.utils.make_chunks(audio, self.chunk_duration)

            for i, chunk in enumerate(chunks):
                if len(chunk) >= self.chunk_duration:
                    chunk.export(os.path.join(self.dataset_path, filename, f"{i}.wav"), format="wav")    
            
    

    def create_dataset(self):
        self.train_data = tf.keras.utils.audio_dataset_from_directory(
            self.dataset_path,
            batch_size = 8,
            shuffle = True,
            validation_split = 0.2,
            subset = "training",
            output_sequence_length = 48000,
            ragged = False,
            label_mode = "categorical",
            labels = "inferred",
            sampling_rate = None,
            seed = 59
        )

        self.validation_data = tf.keras.utils.audio_dataset_from_directory(
            self.dataset_path,
            batch_size = 8,
            shuffle = True,
            validation_split = 0.2,
            subset = "validation",
            output_sequence_length = 48000,
            ragged = False,
            label_mode = "categorical",
            labels = "inferred",
            sampling_rate = None,
            seed = 59

        )
        

    def create_model(self):
        self.mymodel = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size = 80,strides = 16, activation = "relu", input_shape = (48000, 1)),
            tf.keras.layers.MaxPooling2D(4),

            tf.keras.layers.Conv1D(16, kernel_size = 3, activation = "relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(4),

            tf.keras.layers.Conv1D(16, kernel_size = 3,activation = "relu"),
            tf.keras.layers.MaxPooling1D(4),

            tf.keras.layers.Conv1D(32, kernel_size = 3,activation = "relu"),
            tf.keras.layers.MaxPooling1D(4),

            tf.keras.layers.Conv1D(32, kernel_size = 3,activation = "relu"),
            tf.keras.layers.MaxPooling1D(4),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(17, activation = "softmax")
        ])


    def train(self):

        self.mymodel.compile(tf.keras.optimizers.Adam(learning_rate= 0.0001),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"])
        
        checkpoint_filepath = 'model/weights.h5'

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.output = self.mymodel.fit(self.train_data,
        validation_data = self.validation_data,
        epochs = 100,
        callbacks=[model_checkpoint_callback])


    def evaluate(self):
        plt.plot(self.output.history['loss'])
        plt.plot(self.output.history['accuracy'])
        plt.xlabel('epoch')
        plt.legend(['loss', 'accuracy'], loc='upper right')
        plt.title('Loss and accuracy')
        plt.show()


    def predict(self, model, wav_file):
        model = load_model(model)
        waveform, _ = librosa.load(wav_file, sr=None)

        desired_length = 48000
        resized_waveform = librosa.util.fix_length(waveform, size = desired_length)

        input_data = np.expand_dims(resized_waveform, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)

        prediction = model.predict(input_data)
        label = np.argmax(prediction)

        labels = os.listdir(self.dataset_path)
        pred = labels[label]

        print(pred + "😍")



if __name__ == "__main__":
    
    raw_data_path = "raw_data"
    

    audioclassifier = AudioClassifier()

    parser = ArgumentParser()
    parser.add_argument("--raw_data_path", type = str, default = "raw_data", help = "set the path of raw data")
    parser.add_argument("--min_silence_len", type = int, default = 2000, help = "enter min silence len in milisecond")
    parser.add_argument("--silence_thresh" , type = int, default = -45, help = "enter the silence_thresh")
    parser.add_argument("--chunk_duration", type = int, default = 1000, help = "enter the chunk duration in milisecond")

    parser.add_argument("--model", type = str, default='model/weights.h5', help = "the path of .h5 file")
    parser.add_argument("--wav_file", type = str, default = "test_files/test.wav", help = "the path of wav file to inference")
    
    args = parser.parse_args()
    
    audioclassifier.preprocss(args.raw_data_path, args.min_silence_len, args.silence_thresh, args.chunk_duration)
    audioclassifier.create_dataset()
    audioclassifier.create_model()
    audioclassifier.train()
    audioclassifier.evaluate()
    audioclassifier.predict(args.model, args.wav_file)