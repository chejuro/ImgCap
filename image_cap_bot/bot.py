import urllib

import telebot
from flask import Flask, request
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from imgcap_models import VeryModel

TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN, threaded=False)

backbone = models.inception_v3(pretrained=True)
backbone.fc = nn.Identity()
backbone = backbone.eval()

# Load trained model
model_pt = VeryModel()
model_state_path = 'image_cap_bot/data/best_checkpoint_model.pth'
model_pt.load_state_dict(torch.load(model_state_path, map_location='cpu'))
model_pt.eval()


def image_center_crop(image):
    h, w = image.shape[0], image.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2

    return image[pad_top:h - pad_bottom, pad_left:w - pad_right, :]


def preprocess_image(img_name):
    image = cv2.imread(img_name)
    image = image_center_crop(image)
    image = cv2.resize(image, (299, 299))
    image = np.transpose(image, (2, 0, 1))
    image = (image / 255.).astype(np.float32)

    return image


def generate_caption(img_name):
    image = preprocess_image(img_name)
    torch_image = torch.Tensor(image[None, ...]).to('cpu')
    image_embed = backbone(torch_image).ravel()
    caption = ' '.join(model_pt.predict(image_embed, t=1)[1:-1])

    return caption


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Wagwan what?")


# image command handler
@bot.message_handler(content_types=['document'])
def handle_image_messages(message):
    print("Got image")
    print(message)

    document_id = message.document.file_id
    file_info = bot.get_file(document_id)
    filename = file_info.file_path.split('/')[1]
    urllib.request.urlretrieve(f'http://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}',
                               os.path.join('image_cap_bot/images/', filename))

    caption = generate_caption(os.path.join('image_cap_bot/images/', filename))

    bot.send_message(message.chat.id, caption)


# photo command handler
@bot.message_handler(content_types=['photo'])
def handle_photo_messages(message):
    print("Got photo")
    print(message)

    document_id = message.photo[-1].file_id
    file_info = bot.get_file(document_id)
    filename = file_info.file_path.split('/')[1]
    urllib.request.urlretrieve(f'http://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}',
                               os.path.join('image_cap_bot/images/', filename))

    caption = generate_caption(os.path.join('image_cap_bot/images/', filename))

    bot.send_message(message.chat.id, caption)


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    bot.reply_to(message, "Send me an image, pls")


# set webhook
server = Flask(__name__)


@server.route('/' + TOKEN, methods=['POST'])
def get_message():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def web_hook():
    bot.remove_webhook()
    bot.set_webhook(url=os.getenv('HEROKU_URL') + TOKEN)
    return "!", 200


# application entry point
if __name__ == '__main__':
    server.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8443)))
