import requests
import time



while True:
    time.sleep(1)
    headers = {'Content-Type': 'application/octet-stream'}
    t1 = time.time()
    response = requests.post(url='http://127.0.0.1:8085', headers=headers, data=[])
    t2 = time.time()
    time_res = t2-t1
    print(response, time_res)

