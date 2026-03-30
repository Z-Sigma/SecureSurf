import pytest
from src.feature_engineering import get_url_length, count_letters, count_digits

def test_get_url_length():
    assert get_url_length("google.com") == 10
    assert get_url_length("http://google.com") == 17

def test_count_letters():
    assert count_letters("google123") == 6
    assert count_letters("12345") == 0

def test_count_digits():
    assert count_digits("google123") == 3
    assert count_digits("google") == 0

def test_special_chars():
    from src.feature_engineering import count_special_chars
    assert count_special_chars("google.com/test?q=1") == 4 # . / ? =
