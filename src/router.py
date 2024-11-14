from src.preprocessing import *
from src.topic_engine import *
from src.sentiment_engine import *


def _preprocessing(docs: pd.DataFrame | list[str], mode: str, custom_pipeline: dict[str:bool] | None = None):
    if isinstance(docs, pd.DataFrame):
        if docs.iloc[:, 0].dtype != "str" and len(docs.columns) != 1:
            raise ValueError("The input data must be a DataFrame with a single column of strings")
        original = docs.iloc[:, 0].tolist()

    else:
        original = docs

    preprocessing_pipelines = {
        "Topic": {
            "clean_flag": True,
            "delete_stop_words_flag": True,
            "lemmatize_flag": True,
            "keep_only_list": ['NOUN', 'ADJ', 'VERB', 'ADV'],
            "stem_flag": False
        },
        "Sentiment": {
            "clean_flag": True,
            "delete_stop_words_flag": True,
            "lemmatize_flag": True,
            "keep_only_list": [],
            "stem_flag": False
        }
    }
    if mode == "custom" and custom_pipeline:
        x_pipeline = custom_pipeline

    elif mode in preprocessing_pipelines.keys():
        x_pipeline = preprocessing_pipelines[mode]

    else:
        raise ValueError("The mode must be either 'Topic', 'Sentiment' or 'custom'")

    return preprocess(original, **x_pipeline)["Edited"]


def make_topics(docs: pd.DataFrame | list[str], preprocessing_flag: bool = False,
                topic_model_path: str = "../models/good_one.pkl",
                topics_path: str = "../models/topics/good_one.json") -> pd.DataFrame:
    if isinstance(docs, pd.DataFrame):
        if docs.iloc[:, 0].dtype != "str" and len(docs.columns) != 1:
            raise ValueError("The input data must be a DataFrame with a single column of strings")
        original = docs.iloc[:, 0].tolist()
    else:
        original = docs

    if not topics_path.endswith(".json"):
        raise ValueError("The topics path must be a json")

    if preprocessing_flag:
        edited = _preprocessing(original, "Topic")
    else:
        edited = original

    topics = json.load(open(topics_path, encoding="utf-8"))

    model = Tengine(original, edited, topics)
    model.load_model(topic_model_path)

    model.fit_transform()

    model.reduce_outliers()

    result = model.topics_names_to_df()["Topic"]
    if isinstance(docs, pd.DataFrame):
        docs["Topic"] = result
        return docs
    else:
        return pd.DataFrame({
            "Doc": original,
            "Topic": result
        })


def make_sentiment(docs: pd.DataFrame | list[str], preprocessing_flag: bool = False,
                   sentiment_model_path: str = "../models/rubert-tiny2-russian-sentiment") -> pd.DataFrame:
    if isinstance(docs, pd.DataFrame):
        if docs.iloc[:, 0].dtype != "str" and len(docs.columns) != 1:
            raise ValueError("The input data must be a DataFrame with a single column of strings")
        original = docs.iloc[:, 0].tolist()
    else:
        original = docs

    if preprocessing_flag:
        edited = _preprocessing(original, "Sentiment")
    else:
        edited = original

    model = Sengine(model_path=sentiment_model_path)
    predicted = [model.predict(doc) for doc in edited]

    if isinstance(docs, pd.DataFrame):
        docs["Sentiment"] = predicted
        return docs
    else:
        return pd.DataFrame({
            "Doc": original,
            "Sentiment": predicted
        })


def modelling_pipeline(docs: pd.DataFrame | list[str], preprocessing_flag: bool = False,
                       pipeline: list[str] = ["Topic", "Sentiment"]) -> pd.DataFrame:
    if isinstance(docs, pd.DataFrame):
        if docs.iloc[:, 0].dtype != "str" and len(docs.columns) != 1:
            raise ValueError("The input data must be a DataFrame with a single column of strings")
        original = docs.iloc[:, 0].tolist()
    else:
        original = docs
        docs = pd.DataFrame(
            {"Doc": docs}
        )

    if not pipeline:
        raise ValueError("The pipeline must contain at least one element")
    if preprocessing_flag:
        edited = _preprocessing(original, "Topic")
    else:
        edited = docs
    pipeline_kwargs = {
        "Topic": {"func": make_topics,
                  "kwargs": {"topic_model_path": "../models/good_one.pkl",
                             "topics_path": "../models/topics/good_one.json"}
                  },
        "Sentiment": {
            "func": make_sentiment,
            "kwargs": {"sentiment_model_path": "../models/rubert-tiny2-russian-sentiment"}
        }
    }
    for age in pipeline:
        if age not in pipeline_kwargs.keys():
            raise ValueError("The pipeline must contain only 'Topic' or 'Sentiment'")
        print(f"Processing {age}...")
        func, kwargs = pipeline_kwargs[age].values()
        docs[age] = func(docs=edited, preprocessing_flag=False, **kwargs)[age]

    return docs
