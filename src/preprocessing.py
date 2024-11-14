import re
import nltk
import pandas as pd
import pymorphy3
from spacy.util import load_model_from_path
from pathlib import Path
from nltk.stem import SnowballStemmer
from nltk.data import find
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='spacy.util')

try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

STOPWORDS_PATH = Path("../data/stop-ru.txt")
TAGGER_MODEL_PATH = Path("../data/ru_core_news_sm-3.1.0/ru_core_news_sm/ru_core_news_sm-3.1.0")

_morph = pymorphy3.MorphAnalyzer()
_nlp = load_model_from_path(TAGGER_MODEL_PATH)
_stemmer = SnowballStemmer("russian")

with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    _stopwords = [line.strip() for line in f]

_flags = {
    "clean": False,
    "delete_stopwords": False,
    "lemmatize": False,
    "keep_only_nouns": False,
    "stem": False
}

def clear_text(text: str) -> str:
    clean_text = re.sub('<[^<]+?>', '', text)
    clean_text = clean_text.lower()
    clean_text = clean_text.replace("i", "и")

    abbreviations = {
        "цмф": "цинтрийские масла феникса",
        "суоч": "система управления операционными чарами",
        "змс": "зачарованная маслостанция",
        "нсс": "назаирская система слива",
        "лдс": "лекарский договор страхования",
        "цист": "цинтрийский стандарт",
        "сокг": "система оплаты карточкой гильдии"}
    for abbr, full_form in abbreviations.items():
        clean_text = clean_text.replace(abbr, full_form)

    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'[^\u0400-\u04FF\s]', '', clean_text)
    return clean_text


def delete_stopwords(text: str) -> str:
    return " ".join([i for i in word_tokenize(text) if i not in _stopwords])


def lemmatize(text: str) -> str:
    return " ".join([_morph.parse(i)[0].normal_form for i in text.split()])


def keep_only(text: str, parts: list[str]) -> str:
    if parts:
        doc = _nlp(text)
        return " ".join([token.text for token in doc if token.pos_ in parts])
    return text


def stem(text: str) -> str:
    return " ".join([_stemmer.stem(i) for i in text.split()])


def pipeline(text: str, clean_flag: bool = True, delete_stop_words_flag: bool = True,
             lemmatize_flag: bool = True, keep_only_list: list[str] = ['NOUN', 'ADJ', 'VERB', 'ADV'],
             stem_flag: bool = False) -> str:
    _flags.update({
        'clean': clean_flag,
        'delete_stopwords': delete_stop_words_flag,
        'lemmatize': lemmatize_flag,
        'keep_only_nouns': keep_only_list,
        'stem': stem_flag
    })

    steps = [
        (clean_flag, clear_text),
        (delete_stop_words_flag, delete_stopwords),
        (lemmatize_flag, lemmatize),
        (len(keep_only_list) > 0, lambda text: keep_only(text, keep_only_list)),
        (stem_flag, stem)
    ]

    for flag, func in steps:
        if flag:
            text = func(text)

    return text


def preprocess(docs: list[str], clean_flag: bool = True, delete_stop_words_flag: bool = True,
               lemmatize_flag: bool = True, keep_only_list: list[str] = ['NOUN', 'ADJ', 'VERB', 'ADV'],
               stem_flag: bool = False) -> pd.DataFrame:
    df = pd.DataFrame({
        "Original": docs,
        "Edited": docs
    })

    df["Edited"] = df["Original"].apply(lambda x: pipeline(x,
                                                           clean_flag=clean_flag,
                                                           delete_stop_words_flag=delete_stop_words_flag,
                                                           lemmatize_flag=lemmatize_flag,
                                                           keep_only_list=keep_only_list,
                                                           stem_flag=stem_flag))

    return df
