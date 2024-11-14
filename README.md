# Решение для кейса DS в нефтяном ритейле

**Выполнил Тупицин Станислав Павлович**

В качестве решения для кейса был создан ряд модулей, а также пример их компановки, которые позволяют производить
тематическое(и не только) моделирование текстовых данных.

**Реализованные модули:**
1. `preprocessing.py` - модуль предобработки текстовых данных
2. `topic_engine.py` - модуль тематического моделирования
3. `sentiment_engine.py` - модуль анализа тональности текстов
4. `router.py` - пример компановки модулей

## Схема работы модулей следующая:
![img](https://sun9-55.userapi.com/impg/tbNKetByORozbM5uzcPZ7hQ8k9QQTltSjDs9_g/9t3yrqaz-58.jpg?size=511x371&quality=96&sign=b8c973ceca77472f203465b670fee279&type=album)

## Используемый стек:
    Python - 3.10.14

    numpy=1.26.4
    pandas=2.2.2
    bertopic=0.16.4
    sentence-transformers=3.2.1
    nltk=3.9.1
    pymorphy3=2.0.2
    spacy=3.7.6
    transformers=4.46.2
    setuptools=75.4.0
    plotly=5.24.1
    tqdm=4.67.0

## Числовые характеристики системы

   1. Занимаемый объем: 370 МБ без учета установленных зависимостей
   2. Точность кластеризации при тематическом моделировании - 0.68-0.99 в зависимости от данных документов при стандартной конфигурации
   3. Скорость работы на AMD Ryzen 5 5600H 3.30 GHz, 16 GB RAM - 0.049 с. в среднем на строку данного датасета

