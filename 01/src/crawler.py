import requests
from bs4 import BeautifulSoup
import time

class Product:

    def __init__(self):
        self.title = ''
        self.price = ''
        self.country = ''
        self.variety = ''
        self.altitude = ''
        self.processing = ''
        self.taste = ''
        self.bio = 'Ne'

    def __str__(self):
        return f'{self.title};{self.price};{self.country};{self.variety};{self.altitude};{self.processing};{self.taste};{self.bio}'


class Crawler:
    seed = 'https://www.manucafe.cz/'
    queue = [seed]
    crawled = []
    products = []

    def process(self):
        """
        Crawl all pages in queue and process them
        """

        # While queue is not empty
        while self.queue:
            if len(self.crawled) % 50 == 0 and len(self.crawled)>0:
                print('Sleeping...')
                time.sleep(60)

            page = self.queue.pop()

            if page not in self.crawled:
                self.crawled.append(page)
            else:
                continue
            try:
                # print('Crawled:'+page)
                raw = requests.get(page, headers={'User-Agent': 'Testing_crawler'})
                raw.encoding = 'utf-8'
                source = raw.text
                soup = BeautifulSoup(source, "html5lib")

                # Get other links
                self._get_links(soup)

                # Parse
                self._parse(soup)

            except Exception as e:
                # Dont need catch expetions
                pass

    def _parse(self, soup):
        """
        Get all infromations from page
        Price, title, taste, ...
        :param soup:
        :return:
        """
        product = Product()
        type_tag = soup.find("meta", property="og:type")
        type = None
        if type_tag:
            type = type_tag.get("content", None)
        else:
            return
        if not type == 'product':
            return

        # Title
        name = soup.find('h1')
        product.title = name.text.strip()
        par_list = soup.body.findAll('div', attrs={'class': 'parameter-row'})

        if not par_list:
            return
        # Other atributes
        for par in par_list:
            line = str(par.text).replace('\n', '')
            line_split = line.split()
            key = line_split[0]

            if key == 'Země':
                product.country = ' '.join(line_split[2:])
            elif key == 'Odrůda:':
                product.variety = ' '.join(line_split[1:])
            elif key == 'Nadm.':
                product.altitude = ' '.join(line_split[2:])
            elif key == 'Zpracování:':
                product.processing = ' '.join(line_split[1:])
            elif key == 'Chuť:':
                product.taste = ' '.join(line_split[1:])
            elif key == 'BIO:':
                product.bio = ' '.join(line_split[1:])

        product.price = self._parse_price(soup.text)
        self.products.append(product)

    def _parse_price(self, text):
        """
        Dummy function for price extraction
        :param text: page in plaintext
        :return: Price for 1000g or nothing
        """
        found = False
        for line in text.splitlines():
            if found and 'Kč s DPH' in line:
                return ''.join(line.split()[0:-3])
            if '1000g' in line:
                found = True
            if 'Malá balení:' in line:
                return ''

    def _get_links(self, soup):
        """
        Find all links in  page and add them into queue
        Ignore pictures, links with filter, etc...
        :param soup:
        """
        links = soup.findAll('a', href=True)
        for link in links:
            if '?filtr' in link['href'] or '.jpg' in link['href'] or '.png' in link['href'] or '?handlelogin' in link[
                'href']:
                continue

            self.queue.append(self.seed + link['href'])

    def write_CSV(self, file):
        """
        Write all products into CSV file
        :param file: Output file name
        """
        with open(file, 'w', encoding='utf-8') as f:
            f.write(f'Title;Price(Kč/1000g);Country;Variety;Altitude;Processing;Taste;BIO\n')
            for product in self.products:
                f.write(str(product))
                f.write('\n')


crawler = Crawler()
crawler.process()
crawler.write_CSV('output.csv')
