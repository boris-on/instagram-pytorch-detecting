from pyInstagram.instagram import Media, WebAgent
import requests
import json
import sqlite3
from script_for_net import is_it
import config
def make_new_db(tag, page_count):

    conn = sqlite3.connect("pic_db")
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS PICS')
    cursor.execute("""CREATE TABLE PICS
                (id INTEGER PRIMARY KEY,  URL text, ID_inst text)""")
    conn.commit()

    end_cursor = ''
    agent = WebAgent()

    for i in range(0, page_count):
        url = config.find_by_tag.format(tag, end_cursor)
        r = requests.get(url)
        data = json.loads(r.text)
        end_cursor = data['graphql']['hashtag']['edge_hashtag_to_media']['page_info']['end_cursor']
        edges = data['graphql']['hashtag']['edge_hashtag_to_media']['edges']
        for item in range(len(edges)):

            url = str(Media(edges[item]['node']['display_url']))
            nametopost = str(edges[item]['node']['owner']['id'])

            img_data = requests.get(url).content
            with open('dataset/sample.jpg', 'wb') as handler:
                handler.write(img_data)

            if is_it() == "flower":
                url_and_name = [url, nametopost]
                cursor.execute("insert into PICS (URL, ID_inst) values (?, ?)", url_and_name)
                conn.commit()

        print('page downloaded')
    conn.close()