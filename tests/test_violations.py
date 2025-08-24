from utils.violations import generate_negative


def test_generate_negative_injects_hashtags_and_url_and_name():
    chosen = "This is a concise post about AI."
    constraints = {
        "length": "Up to 50 characters",
        "emoji_usage": "none",
        "hashtag_limit": 1,
        "allow_urls": False,
        "allow_names": False,
    }
    neg = generate_negative(chosen, constraints)
    # Expect at least a URL-like string and multiple hashtags
    assert "http" in neg or "https" in neg
    assert neg.count("#") >= 2

