
"""
Негативы в MNRL попадают только в docs_all, то есть всегда в столбцы матриц похожести. Строками везде выступают либо local_queries, 
либо local_docs = docs[0] — то есть одни позитивы. Ни в одном из четырёх directions негатив не является объектом, которому приписана 
метка. Отсюда: два одинаковых негатива не могут породить ложную метку — это структурно невозможно, а не «обычно безопасно».

Единственный реальный эффект — такой негатив дважды попадает в знаменатель softmax и получает вдвое больший вес в градиенте. 
При hardness_mode штраф ему тоже начисляется дважды. Это лёгкое перевзвешивание конкретного негатива, а не порча разметки. 
Так что да, коллизия безопасна.

Ваше правило даже строже, чем нужно. Раз негативы никогда не строки, безопасна не только пара негатив-негатив, но и якорь-негатив. 
Опасны ровно три случая: anchor==anchor, positive==positive, positive==negative. Я оставил проверку якоря против негативов (запрос 
и пассаж совпадают исчезающе редко, так что она ничего не стоит), но при желании её тоже можно снять.

Корень проблемы может быть не там. У mine_hard_negatives дефолт output_format="triplet" — значит одна пара (anchor, positive) превращается в 
num_negatives отдельных строк. При num_negatives=3 у вас 1917 строк, где каждый якорь и каждый позитив встречаются трижды. 
Эти дубликаты NoDuplicatesBatchSampler откладывает всегда, и ваша поблажка на них не распространяется — она лечит только совпадение 
негативов у разных запросов.

совпадения негативов между запросами часты для датасетов с намайнеными hard-негативами из маленьких корпусов 

Ради чего всё это. Сэмплер конфликтные строки не выбрасывает, а откладывает в следующие батчи (sampler.py:588), и __len__ честно назван оценкой сверху. 
Данные не теряются, но хвостовые батчи вырождаются в мелкие, а для MNRL размер батча — это буквально количество негативов. В этом и выигрыш от 
послабления: меньше откладываний, полнее батчи.

Почему нельзя обойтись маленьким переопределением:
get_sample_values возвращает плоский set значений всех столбцов, а _has_overlap — обычный isdisjoint. Принадлежность столбцу теряется до 
всякой проверки. И выразить это через один set нельзя в принципе: негатив должен конфликтовать с позитивом того же текста, но не конфликтовать 
с негативом того же текста — асимметричное отношение, а членство в множестве симметрично. Нужны два множества, а значит __iter__ придётся 
переписать (~50 строк, логика отложенного связного списка сохранена дословно).

Два ограничения, которые я заложил осознанно: precompute_hashes=True запрещён явной ошибкой (в hash-пути столбцы схлопнуты в плоский массив, 
восстановить принадлежность нельзя), а негативные столбцы определяются по префиксу negative — под triplet и n-tuple подходит, для своих имён 
есть параметр negative_columns.

NoDuplicates    : [[2, 3, 4], [0], [1]]
ExceptNegatives : [[2, 3, 4], [0, 1]]
rows 0+1 together: True     <- общий негатив, разрешено
rows 0+2 together: False    <- общий якорь
rows 0+3 together: False    <- общий позитив
rows 0+4 together: False    <- позитив одного == негатив другого
"""

from collections.abc import Iterator

import numpy as np
import torch

from sentence_transformers.base.sampler import (
    NoDuplicatesBatchSampler,
    _EXCLUDE_DATASET_COLUMNS,
    _sample_value_str,
)


class NoDuplicatesExceptNegativesBatchSampler(NoDuplicatesBatchSampler):
    """
    Like NoDuplicatesBatchSampler, but two rows may share a batch when their only
    common value sits in a negative column.

    In MultipleNegativesRankingLoss negatives only ever occupy columns of the
    similarity matrices, never rows, so a repeated negative cannot produce a wrong
    label: it just appears twice in the softmax denominator. Anchor/anchor,
    positive/positive and positive/negative collisions still split the rows apart.
    """

    def __init__(self, *args, negative_columns: list[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.precompute_hashes:
            raise ValueError(
                "precompute_hashes=True is not supported here: the hash path erases column identity."
            )
        columns = [c for c in self.dataset.column_names if c not in _EXCLUDE_DATASET_COLUMNS]
        if negative_columns is None:
            negative_columns = [c for c in columns if c.startswith("negative")]
        if not negative_columns:
            raise ValueError(f"No negative columns found among {columns}.")
        self.negative_columns = set(negative_columns)

    def _split_values(self, index: int) -> tuple[set[str], set[str]]:
        labeled: set[str] = set()
        negatives: set[str] = set()
        for key, value in self.dataset[index].items():
            if key in _EXCLUDE_DATASET_COLUMNS:
                continue
            target = negatives if key in self.negative_columns else labeled
            target.add(_sample_value_str(value))
        return labeled, negatives

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        num_rows = len(self.dataset)
        if num_rows == 0:
            return

        index_dtype = torch.int32 if num_rows <= np.iinfo(np.int32).max else torch.int64
        remaining_indices = torch.randperm(num_rows, generator=self.generator, dtype=index_dtype).numpy()

        position_dtype = np.int32 if num_rows + 1 <= np.iinfo(np.int32).max else np.int64
        next_positions = np.arange(1, num_rows + 1, dtype=position_dtype)
        next_positions[-1] = -1
        head_position = 0

        while head_position != -1:
            batch_labeled: set[str] = set()
            batch_negatives: set[str] = set()
            batch_indices: list[int] = []
            current_position = head_position
            previous_position = -1
            full_batch = False
            while current_position != -1:
                next_position = int(next_positions[current_position])
                index = int(remaining_indices[current_position])
                labeled, negatives = self._split_values(index)

                # An anchor/positive must not clash with anything already in the batch.
                # A negative must not clash with an anchor/positive, but may repeat a
                # negative that is already there.
                if (
                    not labeled.isdisjoint(batch_labeled)
                    or not labeled.isdisjoint(batch_negatives)
                    or not negatives.isdisjoint(batch_labeled)
                ):
                    # Defer conflicting samples to later batches instead of reordering them.
                    previous_position = current_position
                    current_position = next_position
                    continue

                batch_indices.append(index)
                if previous_position == -1:
                    head_position = next_position
                else:
                    next_positions[previous_position] = next_position

                if len(batch_indices) == self.batch_size:
                    full_batch = True
                    yield batch_indices
                    break

                batch_labeled.update(labeled)
                batch_negatives.update(negatives)
                current_position = next_position

            if not full_batch:
                if not self.drop_last:
                    yield batch_indices