from pathlib import Path
import argparse

from datasets import load_dataset

from hybrid_retrieval.util.io import read_json, read_config


# ruff: noqa: C408
ARGS_DEFAULT = dict(
    train_config =    dict(default="configs/train/train_config.yaml"), 
    data_dir =        dict(default="./data/dataset", help="relative path for the other data files"),
    inv_qrels =       dict(default="inv_qrels.json", help="inverted qrels: passage id → queries ids"),
    passages_splits = dict(default="passages_splits2.json", help="dict split (train/test) → queries of split"),
    passages =        dict(default="passages.json", help="passages texts and ids (column format)"),
    queries =         dict(default="queries.json", help="queries texts and ids (column format)"),
    train_dataset =   dict(default="hard_negatives/ds_h2.json", help="training dataset in n-tuple format"),
    out_dir =         dict(default="result"),
)


def parse_args(args_default):
    parser = argparse.ArgumentParser()
    for name, v in args_default.items():
        parser.add_argument(f"--{name}", **{"type": type(v["default"]), **v})
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
