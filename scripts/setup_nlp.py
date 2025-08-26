import sys

try:
    import spacy
except Exception:
    spacy = None

try:
    import nltk
except Exception:
    nltk = None


def main() -> int:
    # Download spaCy model if spaCy is available
    if spacy is not None:
        try:
            from spacy.cli import download as spacy_download
            # Prefer small model for portability
            spacy_download("en_core_web_sm")
            print("Downloaded spaCy model: en_core_web_sm")
        except Exception as e:
            print(f"Skipping spaCy download: {e}")
    else:
        print("spaCy not installed; skipping spaCy model download")

    # Download NLTK resources if NLTK is available
    if nltk is not None:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            print("Downloaded NLTK resources: punkt, stopwords, vader_lexicon")
        except Exception as e:
            print(f"Skipping NLTK downloads: {e}")
    else:
        print("NLTK not installed; skipping NLTK resource download")

    return 0


if __name__ == "__main__":
    sys.exit(main())

