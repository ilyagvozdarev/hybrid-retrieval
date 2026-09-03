"""Разбор CLI-аргументов и чтение данных, общие для всех скриптов обучения."""

from pathlib import Path
import argparse

from datasets import load_dataset

from ir_pipeline.util.io import read_json, read_config


ARGS_DEFAULT = dict(
    train_config="configs/train/train_config.yaml",
    data_dir="../data/dataset",
    inv_qrels="inv_qrels.json",
    passages_splits="passages_splits2.json",
    passages="passages.json",
    queries="queries.json",
    train_dataset="hard_negatives/ds_h2.json",
    out_dir="result",
)


def parse_args(args_default):
    parser = argparse.ArgumentParser()
    for name, default in args_default.items():
        parser.add_argument(f"--{name}", type=type(default), default=default)
    return parser.parse_args()


def read_data(args):
    """
    читает конфиг обучения и данные, указанные в args (dict или Namespace).

    Пути к данным берутся относительно data_dir, путь к train_config — относительно cwd.
    Возвращает копию args, в которой появились:
        train_config    dict из yaml/json
        train_dataset   datasets.Dataset для обучения
        passages        {passage_id: текст} — весь корпус
        queries_eval    {query_id: текст} — только тестовый сплит
        qrels_eval      {query_id: [passage_id]} — только тестовый сплит
    """
    args = dict(vars(args) if isinstance(args, argparse.Namespace) else args)
    data_dir = Path(args["data_dir"])

    args["train_config"] = read_config(args["train_config"])

    inv_qrels = read_json(data_dir / args["inv_qrels"])
    passages_splits = read_json(data_dir / args["passages_splits"])
    queries = read_json(data_dir / args["queries"])
    passages = read_json(data_dir / args["passages"])

    queries = {id: q for q, id in zip(queries["query"], queries["id"])}
    passages = {id: p for p, id in zip(passages["passage"], passages["id"])}

    # у каждого запроса только 1 релевантный пассаж
    qrels_eval = {
        qid: [pid]
        for pid, qids in zip(*inv_qrels.values())
        for qid in qids
        if pid in passages_splits["test"]
    }

    args["passages"] = passages
    args["qrels_eval"] = qrels_eval
    args["queries_eval"] = {qid: queries[qid] for qid in qrels_eval}

    args["train_dataset"] = load_dataset(
        "json", data_files=str(data_dir / args["train_dataset"])
    )["train"]

    print(f"\n    qrels_eval: {len(qrels_eval)}\n    queries_eval: {len(args['queries_eval'])}\n")
    print(args["train_dataset"])

    return args
