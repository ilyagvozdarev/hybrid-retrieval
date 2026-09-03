RU_NLTK: frozenset[str] = set(
"""
и
в
во
не
что
он
на
я
с
со
как
а
то
все
она
так
его
но
да
ты
к
у
же
вы
за
бы
по
только
ее
мне
было
вот
от
меня
еще
нет
о
из
ему
теперь
когда
даже
ну
вдруг
ли
если
уже
или
ни
быть
был
него
до
вас
нибудь
опять
уж
вам
ведь
там
потом
себя
ничего
ей
может
они
тут
где
есть
надо
ней
для
мы
тебя
их
чем
была
сам
чтоб
без
будто
чего
раз
тоже
себе
под
будет
ж
тогда
кто
этот
того
потому
этого
какой
совсем
ним
здесь
этом
один
почти
мой
тем
чтобы
нее
сейчас
были
куда
зачем
всех
никогда
можно
при
наконец
два
об
другой
хоть
после
над
больше
тот
через
эти
нас
про
всего
них
какая
много
разве
три
эту
моя
впрочем
хорошо
свою
этой
перед
иногда
лучше
чуть
том
нельзя
такой
им
более
всегда
конечно
всю
между
""".split()
)


from itertools import filterfalse

# filterfalse с привязанным __contains__ держит и цикл, и проверку на стороне C 
# — на каждый токен не исполняется ни одного байткода Python.
def _remove_stop_words(tokens, stops):
    return list(filterfalse(stops.__contains__, tokens))

def _remove_stop_words_batch(docs, stops):
    hit = stops.__contains__
    return [list(filterfalse(hit, doc)) for doc in docs]

def remove_stop_words(texts, stop_words=RU_NLTK):
    if not isinstance(texts, (list, tuple)):
        texts = list(texts)
    method = (
        _remove_stop_words if texts and isinstance(texts[0], str) 
        else _remove_stop_words_batch)       
    return method(texts, frozenset(stop_words))