import re
import string
import pandas as pd
from urllib.parse import urlparse
from tld import get_tld
import ipaddress
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_url_length(url):
    return len(str(url))

def count_letters(url):
    return sum(char.isalpha() for char in str(url))

def count_digits(url):
    return sum(char.isdigit() for char in str(url))

def count_special_chars(url):
    special_chars = set(string.punctuation)
    return sum(char in special_chars for char in str(url))

def has_shortening_service(url):
    pattern = re.compile(r'https?://(?:www\.)?(?:[\w+]+\.)*(\w+)\.\w+')
    match = pattern.search(str(url))
    if match:
        domain = match.group(1)
        common_services = ['bit', 'goo', 'tinyurl', 'ow', 't', 'is', 'cli', 'yfrog', 'migre', 'ff', 'url4', 'twit', 'su', 'snipurl', 'short', 'BudURL', 'ping', 'post', 'Just', 'bkite', 'snipr', 'fic', 'loopt', 'doiop', 'kl', 'wp', 'rubyurl', 'om', 'to', 'bitly', 'cur', 'ity', 'q', 'po', 'bc', 'twitthis', 'u', 'j', 'buzurl', 'cutt', 'yourls', 'x', 'prettylinkpro', 'scrnch', 'filoops', 'vzturl', 'qr', '1url', 'tweez', 'v', 'tr', 'link', 'zip']
        if domain.lower() in common_services:
            return 1
    return 0

def abnormal_url(url):
    try:
        parsed_url = urlparse(str(url))
        hostname = str(parsed_url.hostname)
        if hostname and hostname in str(url):
            return 1
    except:
        pass
    return 0

def secure_http(url):
    return int(urlparse(str(url)).scheme == 'https')

def have_ip_address(url):
    try:
        parsed_url = urlparse(str(url))
        if parsed_url.hostname:
            ipaddress.ip_address(parsed_url.hostname)
            return 1
    except:
        pass
    return 0

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the URL column.
    """
    logger.info("Extracting features...")
    
    df['url_len'] = df['url'].apply(get_url_length)
    df['letters_count'] = df['url'].apply(count_letters)
    df['digits_count'] = df['url'].apply(count_digits)
    df['special_chars_count'] = df['url'].apply(count_special_chars)
    df['shortened'] = df['url'].apply(has_shortening_service)
    df['abnormal_url'] = df['url'].apply(abnormal_url)
    df['secure_http'] = df['url'].apply(secure_http)
    df['have_ip'] = df['url'].apply(have_ip_address)
    
    # Specific character counts
    df['count_dot'] = df['url'].apply(lambda i: i.count('.'))
    df['count_at'] = df['url'].apply(lambda i: i.count('@'))
    df['count_dir'] = df['url'].apply(lambda i: i.count('/'))
    df['count_embed'] = df['url'].apply(lambda i: i.count('//'))
    df['count_percent'] = df['url'].apply(lambda i: i.count('%'))
    df['count_equal'] = df['url'].apply(lambda i: i.count('='))
    df['count_hyphen'] = df['url'].apply(lambda i: i.count('-'))
    
    def get_tld_count(url):
        try:
            res = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
            return len(res.tld.split('.'))
        except:
            return 0
            
    df['count_tld'] = df['url'].apply(get_tld_count)
    
    logger.info("Feature extraction complete.")
    return df

if __name__ == "__main__":
    from src.data_ingestion import load_data
    from src.preprocessing import clean_data
    df = load_data()
    df = clean_data(df.head(100)) # Test on small sample
    df = extract_features(df)
    print(df.columns)
