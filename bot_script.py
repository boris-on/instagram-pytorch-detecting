import telebot
import config
import time
import sqlite3
import requests
import json
import test_beach
from datetime import datetime

channel_name = "@flowers_test"

bot = telebot.TeleBot(config.TOKEN)

local_time = 600

tag = 'flowers'
page_count = 1

def get_random_profile_name(number):
    url = config.url(int(number))
    response = requests.get(url, config.headers)
    data = json.loads(response.content)
    try:
        return data['data']['user']['reel']['user']['username']
    except TypeError:
        return 'None'

def get_post_from_db():
    conn = sqlite3.connect("pic_db")
    count = conn.cursor()
    count.execute("SELECT COUNT(*) FROM PICS")
    num = count.fetchall()[0][0]
    print(num, "posts left")
    if num > 0:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM PICS ORDER BY RANDOM() LIMIT 1;")
        need = cursor.fetchall()
        delete = conn.cursor()
        delete.execute("DELETE FROM PICS where id = {0}".format(need[0][0]))
        conn.commit()
        conn.close()
        return need
    else:
        test_beach.make_new_db(tag, page_count)
        return get_post_from_db()


def post_it():
    start_time = datetime.now()
    values = get_post_from_db()
    url = values[0][1]
    name = get_random_profile_name(values[0][2])
    bot.send_photo(channel_name, url, caption="@{}".format(name))
    print('bot sent some photo of', name, 'just in', datetime.now() - start_time)


def try_to_post():
    try:
        post_it()
    except:
        try_to_post()

while True:
    try_to_post()
    time.sleep(local_time)
