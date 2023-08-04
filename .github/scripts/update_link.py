import requests
from bs4 import BeautifulSoup  
from fake_headers import Headers

class LinkScraper:

    def __init__(self, url, selector, base_url, h2_name, index_file='index.html'):
        self.url = url
        self.selector = selector
        self.base_url = base_url  
        self.index_file = index_file
        self.h2_name = h2_name

    def scrape(self):
        headers = Headers(headers=True).generate()
        response = requests.get(self.url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        for a in soup.select(self.selector):
            if a.select_one('a[href]') is not None:
                href = self.base_url + a.select_one('a[href]')['href']
                txt = a.select_one('a[href]').text
                links.append((href, txt))
                
        return links
    
    def test_links(self):
        links = self.scrape()
        for href, txt in links:
            print(f'Found link: {href} - {txt}')
            
    def update_index(self, links):
        with open(self.index_file, 'r') as file:
            lines = file.readlines()
            
        for i, line in enumerate(lines):
            if self.h2_name == line.strip():
                for href, txt in links:
                    lines.insert(i + 1, f'\t\t\t\t\t\t\t\t<li><a href="{href}">{txt}</a></li>\n')
                break
            
        with open(self.index_file, 'w') as file:
            file.writelines(lines)
        
    def run(self):
        try:
            links = self.scrape()
            self.update_index(links)
        except Exception as e:
            print(f"An error occurred: {e}")
       
# Usage:
scraper1 = LinkScraper(
   'http://theinformation.com', 
   '.title',
   'https://www.theinformation.com/',
   '<!-- The information!!! -->')
scraper1.run()

scraper2 = LinkScraper(
   'https://news.ycombinator.com/', 
   '.titleline',
   '',
   '<!-- Hacker News!!! -->')
scraper2.run()