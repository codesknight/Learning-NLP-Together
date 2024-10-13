import requests
from bs4 import BeautifulSoup

# 目标URL
url = 'https://www.bejson.com/rarechar/'

# 发出HTTP请求获取网页内容
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 查找页面中的特定信息，例如所有的标题
    titles = soup.find_all('a')  # 查找所有h1标签内容

    for index, title in enumerate(titles, 1):
        print(f"Title {index}: {title.get_text()}")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
