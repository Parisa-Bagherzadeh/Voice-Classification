import numpy as np
import os


import telebot
import librosa

from tensorflow.keras.models import load_model

friends_model = load_model('model\weights.h5')
singer_model = load_model('singer/weights/weights.h5')


bot = telebot.TeleBot("")
@bot.message_handler(commands = ['start'])
def send_welcome(message):
    bot.reply_to(message, 'Hello and Welcome!😊\n Send me a voice please!')

  



@bot.message_handler(content_types = ['voice'])
def voice_processing(message):
  myvoice = bot.get_file(message.voice.file_id)
  myfile = bot.download_file(myvoice.file_path)
  voicepath = myvoice.file_path

  with open(voicepath, 'wb') as audio:
    audio.write(myfile)


  waveform, _ = librosa.load(voicepath, sr=None)  
  desired_length = 48000
  resized_waveform = librosa.util.fix_length(waveform, size = desired_length)

  input_data = np.expand_dims(resized_waveform, axis=-1)
  input_data = np.expand_dims(input_data, axis=0)

  model = friends_model

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

      wav_filename = 'output.wav'

      with open(wav_filename, 'wb') as audio:
          audio.write(file)

      waveform, _ = librosa.load(wav_filename, sr=None)  
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