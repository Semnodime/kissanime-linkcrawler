# kissanime-linkcrawler
Dump a list of all available streaming sources of a series from [kissanime](http://kissanime.ru) into a file for later download.

# Development notice
Because the admins have noticed my sucessful circumvention of their captcha system they have adopted by randomly rotating and additionally tagging images with numbers. So simply hashing an image to identify it is no longer enough and I have to rewrite the captcha solver. **Therefore the captcha auto solver is currently not working and I will quickly add an option to manually solve the captchas untill I have adopted the auto captcha solver.**

# Python Version
This program is requiring `python3.3` or later!

# Usage
see `python crawler.py --help`
```
usage: crawler.py [-h] [-t] url

Search for streams of the anime determined by the given url.

positional arguments:
  url          the url of the anime you want to crawl

optional arguments:
  -h, --help   show this help message and exit
  -t, --train  train the captcha solver
```

# Install
- `pip install selenium`
- `pip install bs4`
- `pip install requests`
- Download latest [geckodriver](https://github.com/mozilla/geckodriver/releases) and place it in the project directory.

# TODO
- Implement manual captcha solving option
- Implement support for domains other than kissanime.ru
- Implement search for other streams than only the first one found

# Technical Aspect
This program parses the domain and the anime title from the given url.
Then it unblocks the js-protection by launching a selenium driven browser and obtaining a vaild cookie.
Once the site is unblocked it crawls for all episodes and their streams.

# Captcha Auto-Solver
The program includes automatical captcha detection and an auto-sover.
The captcha solver comes pretrained with >100,000 samples and also learns to solve new captchas on the fly once it faces new captchas.
In case the solver does not recogince new captchas you can train it yourself by uncomm
