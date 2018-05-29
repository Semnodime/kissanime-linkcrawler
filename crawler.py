#!/usr/bin/env python3
"""
This program can be used to crawl kissanime.ru for all streams of all episodes of any given anime url.
"""
import argparse
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
import requests
import re
from bs4 import BeautifulSoup
import json
from urllib.parse import unquote
from json import JSONEncoder
import hashlib
import os.path
from os import makedirs
import sys
import signal
import logging
from multiprocessing.dummy import Pool as ThreadPool


class DelayedKeyboardInterrupt(object):
    def __init__(self):
        self.signal_received = None

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type_, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


cookies = {}
headers = {}
min_occurrences = 2  # Number of times an image has to be listed under an attribute in oder to be valid


class PyJSONEncoder(JSONEncoder):
    def default(self, o):
        return o.__json__()


def hash_string(data: str, limit=8):
    hash_hex = hashlib.sha224(data.encode()).hexdigest()
    return hash_hex[:limit] if limit and limit > 0 else hash_hex


class Captcha:
    class FailedExtraction(Exception):
        pass

    def __init__(self, html_source='', source_url='', attributes=None, images=None):
        if attributes or images:
            assert attributes and images
            self.attributes = attributes
            self.images = images
        elif html_source or source_url:
            assert html_source and source_url
            url_prefix_match = re.search('https?://[^/]+', source_url)
            assert url_prefix_match
            url_prefix = url_prefix_match.group(0)

            """Extract captcha image attributes and image urls (and data) from the html source."""
            parsed_html = BeautifulSoup(html_source, 'html.parser')
            captcha_form = parsed_html.body.find('form', attrs={'action': '/Special/AreYouHuman2'})

            if not captcha_form:
                raise Captcha.FailedExtraction('Could not recognize captcha form in html source')

            attributes = [str(span.text).strip('\r\n ').split(',') for span in captcha_form.find_all('span')]
            self.attributes = [[str(word).strip(' ') for word in attribute] for attribute in attributes]
            self.images = [url_prefix + image.get('src') for image in captcha_form.find_all('img')]

            if not self.attributes:
                raise Captcha.FailedExtraction('Could not extract captcha descriptive text')
            if not self.images:
                raise Captcha.FailedExtraction('Could not extract captcha images')

    def __str__(self):
        return '<Captcha %s %s>' % (str(self.attributes), str(self.images))

    @staticmethod
    def hash_attribute(attribute):
        return ','.join(sorted(attribute))


class CaptchaSolver:
    def __init__(self):
        self.mapping = {}
        self.cache = {}
        self.samples = 0
        self.base_path = 'captcha'
        if not os.path.isdir(self.base_path):
            makedirs(self.base_path)
        self.mapping_path = self.base_path + '/' + 'mapping.json'
        self.cache_path = self.base_path + '/' + 'cache.json'
        self.load()
        self.print_statistic(full=False)

    def load(self):
        """ Load mapping and cache and return success of both"""
        r1 = self.load_cache()
        r2 = self.load_mapping()
        return r1 and r2

    def load_cache(self):
        """ Load the cached hashes of previously fetched resources """
        try:
            with open(self.cache_path, 'rb') as fp:
                data = json.load(fp)
            self.cache = data
            return True

        except FileNotFoundError:
            return False

    def load_mapping(self):
        """ Load the knowledge of a previously trained solver """
        try:
            with open(self.mapping_path, 'r') as fp:
                data = json.load(fp)
            self.mapping = data['mapping']
            self.samples = data['samples']
            return True

        except FileNotFoundError:
            return False

    def save_cache(self):
        """ Save currently available cached hashes of external resources """
        data = self.cache
        with DelayedKeyboardInterrupt():
            with open(self.cache_path, 'w') as fp:
                json.dump(data, fp)

    def save_mapping(self):
        """ Save currently available captcha knowledge """
        data = {'mapping': self.mapping, 'samples': self.samples}
        with DelayedKeyboardInterrupt():
            with open(self.mapping_path, 'w') as fp:
                json.dump(data, fp)

    def save(self):
        """ Save mapping and cache """
        self.save_cache()
        self.save_mapping()
        print('Saved Captcha Solver')

    def learn_from(self, captcha: Captcha):
        for attribute in captcha.attributes:
            key = Captcha.hash_attribute(attribute)
            if key not in self.mapping:
                self.mapping[key] = {}
            for image_url in captcha.images:
                url_hash = hash_string(image_url, limit=None)
                image_hash = self.cache[url_hash] if url_hash in self.cache else self.hash_url_content(image_url)
                if image_hash not in self.mapping[key]:
                    self.mapping[key][image_hash] = 0
                self.mapping[key][image_hash] += 1
        self.samples += 1

    def print_statistic(self, full=False):
        statistic = []
        images = set([])
        for attribute_hash, image_dict in self.mapping.items():
            images = images.union(set(image_dict.keys()))
            highest_img_occurrence = sorted(image_dict.values()).pop()
            statistic.append((highest_img_occurrence, attribute_hash))
        statistic.sort()

        unmapped = []
        for (max_occurrence, attribute_hash) in statistic:
            if max_occurrence < min_occurrences:
                unmapped.append(attribute_hash)

        attributes_count = len(statistic)
        print('Current captcha model contains %d samples, %d attributes (%d unmapped), %d images'
              % (self.samples, attributes_count, len(unmapped), len(images)))

        if full:
            for (max_occurrence, attribute_hash) in statistic:
                print(max_occurrence, '\t', attribute_hash)

    def pick_image(self, attribute, image_hashes):
        """ Attempt to pick the correct image for the given attribute """
        attr_hash = Captcha.hash_attribute(attribute)

        if attr_hash not in self.mapping:
            print('Unknown attribute hash:', repr(attr_hash))
            return None

        possible = [(self.mapping[attr_hash][image_hash], image_hash) for image_hash in image_hashes if image_hash in self.mapping[attr_hash]]
        if len(possible) == 0:
            print('No given image matched the attribute')
            return None

        occurrence, picked = sorted(possible).pop()

        if occurrence < min_occurrences:
            print('Not enough samples of picked image to be considered valid')
            return None
        else:
            return picked

    def solve(self, captcha: Captcha, improve=False):
        """ Returns a list of indices determined by the attributes describing the images OR False on fail """
        print('Solving captcha', captcha)

        if improve:
            print('Improving captcha solver with current sample')
            self.learn_from(captcha)

        image_hashes = [self.hash_url_content(image_url, allow_cache=True) for image_url in captcha.images]

        indices = []
        for attribute in captcha.attributes:
            picked_image = self.pick_image(attribute, image_hashes)
            if not picked_image:
                print('Captcha could not be solved.',
                      'Failed on attribute %s. Training the captcha model might help...' % repr(attribute))
                return False
            indices.append(image_hashes.index(picked_image))
        return indices

    def hash_url_content(self, url, download=False, allow_cache=True):
        url_hash = hash_string(url, limit=None)

        if allow_cache and url_hash in self.cache:
            print('Reading hash from json cache. Fetching/Saving of content skipped.', url)
            return self.cache[url_hash]

        else:
            dir_path = self.base_path + '/cache/'
            if not os.path.isdir(dir_path):
                makedirs(dir_path)
            file_name = '%s.jpg' % url_hash
            file_path = dir_path + file_name

            def file_chunks():
                """ Yield chunks of the file either from disk-cache or fetched data """
                if os.path.isfile(file_path):
                    print('Using cached file from disk instead of fetching external resource', file_path)
                    with open(file_path, 'rb') as file:
                        while True:
                            megabyte = 1024 * 1024
                            chunk = file.read(8 * megabyte)
                            if not chunk:
                                return
                            yield chunk
                else:
                    print('Fetching resource from %r' % url)
                    r = requests.get(url, stream=True, cookies=cookies, headers=headers)
                    if r.status_code != 200:
                        print('Fetching failed with status %d' % r.status_code)
                        return
                    file = open(file_path, 'wb') if download else None
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            if file:
                                file.write(chunk)
                            yield chunk
                    if file:
                        file.close()

            h = hashlib.sha1()
            for chunk in file_chunks():
                h.update(chunk)

        result = h.hexdigest()
        if result == 'da39a3ee5e6b4b0d3255bfef95601890afd80709':
            raise ValueError('Hash implies empty data.')
        self.cache[url_hash] = result
        return result


class CaptchaGenerator:
    """ Dummy class for inheritance """
    def __init__(self):
        raise AssertionError

    def generate(self):
        """ Returns a new Captcha sample"""
        while True:
            yield Captcha()


class KissAnimeCaptchaGenerator(CaptchaGenerator):
    def __init__(self, threads=10, prefetch=20, max_failures=5):
        self.session = requests.session()
        self.threads = threads
        self.prefetch = prefetch
        self.max_failures = max_failures

    def generate(self):
        global cookies
        global headers

        def query_captcha(_dummy_):
            """ Load the website with a captcha and return captcha sample information """
            url = 'http://kissanime.ru/Special/AreYouHuman2?reUrl='
            r = self.session.get(url, cookies=cookies, headers=headers)
            if r.status_code != 200:
                print('Page did not load properly. Status code: %d' % r.status_code)
                return None
            try:
                captcha = Captcha(html_source=r.text, source_url=url)
            except Captcha.FailedExtraction as e:
                print(e)
                return None

            print('Retrieved new captcha sample:', captcha.attributes)
            return captcha

        samples = []
        failures = 0
        fetch_batch = self.prefetch / 2

        while True:
            missing = self.prefetch - len(samples)
            if missing > fetch_batch:
                # Multithread the captcha queries
                with ThreadPool(self.threads) as pool:
                    results = pool.map(query_captcha, [None]*self.prefetch)
                    for sample in results:
                        if not sample:
                            failures += 1
                            if failures >= self.max_failures:
                                return
                        else:
                            samples.append(sample)

            if len(samples) > 0:
                yield samples.pop()


class FileCaptchaGenerator(CaptchaGenerator):
    def __init__(self):
        with open('captchaSamples.json') as file:
            self.samples = json.load(file)

    def generate(self):
        for sample in self.samples:
            yield Captcha(attributes=sample[0], images=sample[1])


def train_captcha_solver(solver: CaptchaSolver, generator: CaptchaGenerator, samples=100, saveeach=100, statistic=False):
    """ Train a captcha solver model by gathering many captcha samples """
    print('Training captcha solver with up to %d samples' % samples)
    count = 0
    for captcha in generator.generate():
        count += 1
        print('Captcha sample %s/%s' % (count, samples))
        try:
            solver.learn_from(captcha)
        except ValueError:
            print('Exception caused a skip of the current captcha', )
            pass
        if saveeach > 0 and count % saveeach == 0:
            solver.save()
            if statistic:
                solver.print_statistic()
        if count == samples:
            solver.save()
            return True

    return False


class Anime:
    class Stream:
        def __init__(self, url: str):
            self.url = url
            self.host = re.search('^https?://(?P<domain>[^?/#]+)]', self.url).group('domain')

        def __str__(self):
            return '<Stream: %s >' % self.url

        def __json__(self):
            return self.url

    class Episode:
        def __init__(self, url):
            self.url = url
            self.number = int(re.search('Episode-(?P<number>\d+)', self.url).group('number'))
            self.streams = []

        def __str__(self):
            return '<Episode %d: %s>' % (self.number, [str(stream) for stream in self.streams])

        def __json__(self):
            return self.streams

    def __init__(self, url):
        match = re.search('(?P<protocol>^https?://)(?P<domain>kissanime\.ru)/Anime/(?P<anime>[^?/]+)', url)
        if not match:
            raise ValueError('Invalid Anime URL', url)
        self.protocol = match.group('protocol')
        self.domain = match.group('domain')
        self.name = match.group('anime')
        self.url = match.group(0)
        self.episodes = []

    def __str__(self):
        return '<Anime %s with %d episodes>' % (self.name, len(self.episodes))

    def __json__(self):
        episode_dict = {}
        for episode in self.episodes:
            episode_dict[episode.number] = episode
        return episode_dict

    def dump(self):
        filename = '%s-streams.json' % self.name
        answer = None
        if os.path.isfile(filename):
            while answer is None:
                answer = input('Do you want to override existing file %r ? (Y/N)' % filename)
                print(repr(answer))
                if str(answer).lower() in ['y', 'yes']:
                    answer = True
                elif str(answer).lower() in ['n', 'no']:
                    answer = False
                else:
                    print('Answer not recognized')
            if answer is False:
                return

        with open('%s-streams.json' % self.name, 'w') as file:
            json.dump(self, file, cls=PyJSONEncoder)
        print('Stream list written into %r' % filename)


class Crawler:
    """ Crawler that crawls for anime episodes and their streaming sources """
    def __init__(self):
        self.protocol = ''
        self.domain = ''

    @staticmethod
    def crawl(anime: Anime):
        """ Attempting to return a list of all streams of all episodes for the given anime """
        global cookies
        global headers

        print('')
        print('Crawling anime %s' % anime.name)

        fetched_episodes = Crawler.get_episodes(anime)
        if not fetched_episodes:
            return
        anime.episodes += fetched_episodes

        task = 0
        for episode in anime.episodes:
            print('\n')
            task += 1
            for retry in range(5, -1, -1):
                print('TASK %d/%d : %s' % (task, len(anime.episodes), 'Episode-%d' % episode.number))
                s = Crawler.harvest_stream(episode.url)
                if s:
                    episode.streams.append(s)
                    print(s)
                    break
                else:
                    print('Retries left: %d' % retry)
            solver.save()

    @staticmethod
    def set_challenge_cookie(url):
        """ Runs the javascript browser check to get a cookie for browsing the site """
        global cookies
        global headers

        print('Running a browser to complete the js-browser-check')
        options = Options()
        options.add_argument("--headless")
        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.tabs.warnOnClose", False)
        profile.set_preference("dom.disable_beforeunload", True)
        driver = webdriver.Firefox(firefox_options=options, firefox_profile=profile)
        driver_useragent = driver.execute_script('return navigator.userAgent')
        headers = {'User-Agent': driver_useragent}
        driver.get(url)
        while True:
            print('Browser at %s' % driver.current_url)
            cookies = driver.get_cookies()
            cookies = list(filter(lambda cookie: cookie['name'] == 'cf_clearance', cookies))
            if len(cookies) == 1:
                print('Got cookie for challenge completion!')
                cookie = cookies.pop()
                cookies = {cookie['name']: cookie['value']}
                driver.quit()
                return
            else:
                print('Waiting for browser to complete js-challenge')
                time.sleep(5)

    @staticmethod
    def check_blocked(url):
        """ Check if the page displays a 'service not available 503' response """
        global cookies
        global headers

        return requests.get(url, headers=headers, cookies=cookies).status_code == 503

    @staticmethod
    def unblock_site(url):
        """ Attempt to remove crawling restrictions by retrieving a valid session with a js-enabled client """
        if Crawler.check_blocked(url):
            print('The service seems to be blocked by a js-browser-check')
            Crawler.set_challenge_cookie(url)
            print('Retrieved cookie to allow browsing the page:', cookies)

    @staticmethod
    def get_episodes(anime):
        """ Returns a list of full urls each being an episode of the series """
        global cookies
        global headers

        url_path = '/Anime/%s/' % anime
        r = requests.get(anime.url, cookies=cookies, headers=headers)
        if r.status_code != 200:
            print('Page did not load properly. Status code: %d' % r.status_code)
            return None
        else:
            print('Loaded site.')

        episode_matches = re.finditer('href="(?P<urlpath>/Anime/%s/Episode-.+?)"' % anime.name, r.text, flags=re.S)
        episodes = [Anime.Episode(anime.protocol + anime.domain + url_path) for url_path in [match.group('urlpath') for match in episode_matches]]
        return sorted(episodes, key=lambda e: e.number)

    @staticmethod
    def harvest_stream(url):
        """ Attempting to list the streaming source from the given url """
        global cookies
        global headers
        global solver

        print('Fetching streaming sources from %s' % repr(url))

        with requests.session() as session:
            '''
            for name, value in dict(cookies).items():
                session.cookies.set(name=name, value=value, domain='.' + domain, path='/')
                print('Set cookie', name, value, '.' + domain, '/')
            '''
            r = session.get(url, headers=headers, cookies=cookies)
            if r.status_code != 200:
                print('Page did not load properly. Status code: %d' % r.status_code)
                return None
            else:
                print('Loaded site.')

            if 'AreYouHuman2' in r.url:
                print('Detected captcha:', r.url)
                try:
                    captcha = Captcha(html_source=r.text, source_url=r.url)
                except Captcha.FailedExtraction as e:
                    print(e)
                    return None

                indices = solver.solve(captcha, improve=True)
                if not indices:
                    return None

                # Send Captcha Solution
                answer = ','.join([str(index) for index in indices]) + ','

                post_url, re_url = str(r.url).split('?')
                re_url = unquote(re_url).replace('reUrl=', '')
                data = {
                    'answerCap': answer,
                    'reUrl': re_url,
                }

                print('Posting solution %s to %s' % (data, post_url))

                r = session.post(post_url, headers=headers, cookies=cookies, data=data)
                if r.status_code != 200:
                    print('Posting the captcha or any of its redirects failed.', 'Status %s' % r.status_code)
                    print('History:', [(h.status_code, h.url) for h in r.history])
                    return None
                elif 'Wrong answer.' in r.text:
                    print('Solving Captcha failed: Wrong answer.')
                    return None

                print('Captcha appears to be solved!')
            else:
                print('Site does not appear to be protected by a captcha')

            if url not in r.url:
                print('We did NOT land on the page we wanted: %s <<<instead of>>> %s ' % (r.url, url))
                print('History:', [(h.status_code, h.url) for h in r.history])

            if "$('#divContentVideo').html(" not in r.text or 'iframe' not in r.text:
                print('Source html seems weird! (No stream found)')
                print(repr(r.text))
                return None

            match = re.search('<iframe .+?src="(?P<stream>https?://.+?)".+?></iframe>', r.text)
            if not match:
                print('Could not find any stream')
                return None

            stream_url = match.group('stream')
            return stream_url


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Search for streams of the anime determined by the given url.")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("url", type=str, help="the url of the anime you want to crawl")
    group.add_argument("-t", "--train", action="store_true", help='train the captcha solver')
    args = parser.parse_args()

    crawl_url = args.url.strip()
    print(args)
    anime = None
    try:
        anime = Anime(crawl_url)
        solver = CaptchaSolver()

        Crawler.unblock_site(crawl_url)
        if args.train:
            train_captcha_solver(solver, KissAnimeCaptchaGenerator(threads=20, prefetch=100), samples=1000, statistic=True, saveeach=50)
        Crawler.crawl(anime)
        solver.save()
        anime.dump()
    except KeyboardInterrupt as e:
        print('KeyboardInterrupt')
        exit(2)
    except Exception as e:
        print('Exception', e)
        exit(1)

