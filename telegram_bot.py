import numpy as np
import os
import pydub


import telebot
import librosa

from tensorflow.keras.models import load_model

# from model import AudioClassifier 



friends_model = load_model('model/weights.h5')
singer_model = load_model('singer/weights/weights.h5')


bot = telebot.TeleBot("6759479250:AAFeanau_rbEx0nfatlAo9yvMrVDn0n_L0g")
@bot.message_handler(commands = ['start'])
def send_welcome(message):
    bot.reply_to(message, 'Hello and Welcome!😊\n Send me a voice please!')

  

@bot.message_handler(content_types = ['voice'])
def voice_processing(message):
  myvoice = bot.get_file(message.voice.file_id)
  myfile = bot.download_file(myvoice.file_path)
  voicepath = myvoice.file_path

  with open('myvoice.wav', 'wb') as newvoice:
    newvoice.write(myfile)

  voice = pydub.AudioSegment.from_file('myvoice.wav')
  voice = voice.set_sample_width(2)
  voice = voice.set_channels(1)
  chunks = pydub.silence.split_on_silence(voice, min_silence_len = 2000, silence_thresh = -45)
  result = sum(chunks)
  # chunks = pydub.utils.make_chunks(result, 1000)
  result.export("new_voice.wav", format = "wav")


  waveform, _ = librosa.load("new_voice.wav", sr=None)  
  desired_length = 48000
  resized_waveform = librosa.util.fix_length(waveform, size = desired_length)

  input_data = np.expand_dims(resized_waveform, axis=-1)
  input_data = np.expand_dims(input_data, axis=0)

  model = load_model(friends_model)

  prediction = model.predict(input_data)

  label = np.argmax(prediction)

  labels = os.listdir('dataset_voice')
  pred = labels[label]
  bot.reply_to(message, f'You are {pred} 😍')




  @bot.message_handler(commands=['singer'])
  def send_welcome(message):
      bot.reply_to(message, 'Send a wav file of a singer')

  @bot.message_handler(content_types=['document'])
  def handle_document(message):
    if message.document.mime_type.startswith('audio'):

      myfile = bot.get_file(message.document.file_id)
      filepath = myfile.file_path
      file = bot.download_file(filepath)

      with open('singer.wav', 'wb') as newvoice:
        newvoice.write(myfile)

      voice = pydub.AudioSegment.from_file('singer.wav')
      voice = voice.set_sample_width(2)
      voice = voice.set_channels(1)
      chunks = pydub.silence.split_on_silence(voice, min_silence_len = 2000, silence_thresh = -45)
      result = sum(chunks)
      result.export("new_voice.wav", format = "wav")

      waveform, _ = librosa.load("new_voice.wav", sr=None)  
      desired_length = 48000
      resized_waveform = librosa.util.fix_length(waveform, size=desired_length)

      input_data = np.expand_dims(resized_waveform, axis=-1)
      input_data = np.expand_dims(input_data, axis=0)

      model = singer_model

      prediction = model.predict(input_data)

      label = np.argmax(prediction)

      labels = os.listdir('singer_dataset')
      pred = labels[label]
      bot.reply_to(message, f'{pred}😍 ')
      print(f"{pred}😍")


bot.polling()



  

