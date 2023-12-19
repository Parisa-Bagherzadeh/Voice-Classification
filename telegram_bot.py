import numpy as np
import os


import telebot
import librosa

from tensorflow.keras.models import load_model


bot = telebot.TeleBot("6759479250:AAFeanau_rbEx0nfatlAo9yvMrVDn0n_L0g")
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

  model = load_model('model\weights.h5')

  prediction = model.predict(input_data)

  label = np.argmax(prediction)

  labels = os.listdir('dataset_voice')
  pred = labels[label]
  bot.reply_to(message, f'You are {pred} 😍')


bot.polling()