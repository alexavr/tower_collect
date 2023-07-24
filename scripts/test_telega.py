from telegram import Bot

# channel_id="@TowerMSU"
# bot = Bot('6260337300:AAHu1UW21bWpzbihKP_65GK4ubY4P-2tJh0')
channel_id="@TowerPIO"
bot = Bot('5787387775:AAF-yF8yLPtdkXMrmRF_PWZzYHz8nKYl7jo')


bot.send_message(channel_id, "TEST") 

# import requests
# import tower_lib as tl

# TOKEN = "6260337300:AAHu1UW21bWpzbihKP_65GK4ubY4P-2tJh0"
# url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
# print(requests.get(url).json())

# res = tl.notification.send2bot(f"6260337300:{TOKEN}", "-1001538910491", "TEST")

# import requests
# TOKEN = "5787387775:AAF-yF8yLPtdkXMrmRF_PWZzYHz8nKYl7jo"
# chat_id = 269679622
# message = "TEST"
# url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
# print(requests.get(url).json()) # Эта строка отсылает сообщение