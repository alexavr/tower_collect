import requests
TOKEN = "6260337300:AAHu1UW21bWpzbihKP_65GK4ubY4P-2tJh0"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
print(requests.get(url).json())


# import requests
# TOKEN = "5787387775:AAF-yF8yLPtdkXMrmRF_PWZzYHz8nKYl7jo"
# chat_id = 269679622
# message = "TEST"
# url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
# print(requests.get(url).json()) # Эта строка отсылает сообщение