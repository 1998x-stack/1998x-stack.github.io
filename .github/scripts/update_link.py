import requests
from bs4 import BeautifulSoup
from fake_headers import Headers


def update_links():
    headers = Headers(headers=True).generate()
    
    # 发送请求获取网页内容
    response = requests.get("http://theinformation.com", headers=headers)

    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 获取所有的链接和文本
    links = []
    for a in soup.select('.title'):
        if a.select_one('a[href]') is not None:
            href = 'https://www.theinformation.com/' + a.select_one('a[href]')['href']
            txt = a.select_one('a[href]').text
            links.append((href, txt))
    try:
        # 读取index.html文件
        with open('index.html', 'r') as file:
            lines = file.readlines()

        # 在<h1>The information</h1>后面插入链接
        for i, line in enumerate(lines):
            if '<h1>The information</h1>' in line:
                for href, txt in links:
                    lines.insert(i + 1, f'<p><a href="{href}">{txt}</a></p>\n')
                break

        # 将更新后的内容写回index.html文件
        with open('index.html', 'w') as file:
            file.writelines(lines)
    except:
        pass


    try:
        # 读取index.html文件
        with open('../../index.html', 'r') as file:
            lines = file.readlines()

        # 在<h1>The information</h1>后面插入链接
        for i, line in enumerate(lines):
            if '<h1>The information</h1>' in line:
                for href, txt in links:
                    lines.insert(i + 1, f'<p><a href="{href}">{txt}</a></p>\n')
                break

        # 将更新后的内容写回index.html文件
        with open('../../index.html', 'w') as file:
            file.writelines(lines)
    except:
        pass



# 使用错误处理机制
try:
    update_links()
except Exception as e:
    print(f"An error occurred: {e}")
