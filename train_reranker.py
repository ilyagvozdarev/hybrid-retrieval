'''
Running the Script:

```bash
python train_reranker.py \
--model "BAAI/bge-reranker-v2-m3" \
--train_config "configs/train/train_config.yaml" \
--data_dir "./data/dataset" \
--inv_qrels "inv_qrels.json" \
--passages_splits "passages_splits2.json" \
--passages "passages.json" \
--queries "queries.json" \
--train_dataset "hard_negatives/ds_h2.json" \
--out_dir "result" \
--ranking "result/eval/dense/Information-Retrieval_evaluation_ir_evaluator_predictions_cosine.jsonl" \
--ranking_top_k 20
```
'''

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.base.sampler import BatchSamplers
from datasets import load_dataset, Dataset

from hybrid_retrieval import data, training
from hybrid_retrieval.config import merge_configs


ARGS_DEFAULT = dict(
    **data.ARGS_DEFAULT,
    model =         dict(default="BAAI/bge-reranker-v2-m3"),
    ranking =       dict(default="result/eval/dense/Information-Retrieval_evaluation_ir_evaluator_predictions_cosine.jsonl", 
                         help="retriever corpus ranking"),
    ranking_top_k = dict(default=20, help="top-k for reranking")
)

TRAIN_CONFIG_DEFAULT = dict(
    optim=dict(
        gradient_checkpointing=True,
        optim="adamw_torch_fused"
    ),
    training=dict(
        num_train_epochs=2,
        train_batch_size=16,
        learning_rate=1.0e-5,
        warmup_steps=0.1
    ),
    seed=42
)

EVALUATOR_NAME = "reranker_evaluator"
AT_K = 10


def read_data(args, dataset_format='n-tuple'):
    """
    К общим данным добавляет ranking — выдачу ретривера, которую реранкер переупорядочивает.
    """
    args = data.read_data(args)
    passages, queries_eval, qrels_eval = args["passages"], args["queries_eval"], args["qrels_eval"]

    ranking = load_dataset("json", data_files=args["ranking"])["train"]
    ranking_ids = {
        qid: [doc["corpus_id"] for doc in docs][: args["ranking_top_k"]]
        for qid, docs in zip(ranking["query_id"], ranking["results"])
    }

    args["ranking"] = [
        {
            "query_id": qid,
            "query": queries_eval[qid],
            "positive": [passages[pid] for pid in qrels_eval[qid]],
            "documents": [passages[pid] for pid in ranking_ids[qid]],
        }
        for qid in queries_eval
    ]
    print(f"    ranking: {len(args['ranking'])} запросов по {args['ranking_top_k']} документов\n")

    if dataset_format == 'labeled-list':
        columns_doc = [col for col in args["train_dataset"].column_names if col != 'anchor']
        labels = [1 if 'positive' in col else 0 for col in columns_doc]
        train_dataset = [
            {
                'anchor': sample['anchor'],
                'document': [sample[col] for col in columns_doc],
                'labels': labels
            }
            for sample in args["train_dataset"]
        ]
        args["train_dataset"] = Dataset.from_list(train_dataset)

    return args


def train_setup_1():
    from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

    args = read_data(data.parse_args(ARGS_DEFAULT))
    train_config = merge_configs(TRAIN_CONFIG_DEFAULT, args["train_config"])

    model = CrossEncoder(args["model"], device=training.device())  
    loss = CachedMultipleNegativesRankingLoss(model, num_negatives=4, mini_batch_size=8)

    evaluator = CrossEncoderRerankingEvaluator(
        samples=args["ranking"],
        at_k=AT_K,
        name=EVALUATOR_NAME,
        # при обучении добавляем позитивы, чтобы оценивать только реранкер для выбора чекпоинта
        always_rerank_positives=True,
    )

    training.run_training(
        model=model,
        loss=loss,
        evaluator=evaluator,
        train_dataset=args["train_dataset"],
        train_config=train_config,
        trainer_cls=CrossEncoderTrainer,
        training_args_cls=CrossEncoderTrainingArguments,
        training_args_kwargs=dict(
            # у cross-encoder'а нет similarity_fn: метрики вида {name}_{metric}
            metric_for_best_model=f"{EVALUATOR_NAME}_ndcg@{AT_K}",
            batch_sampler=BatchSamplers.NO_DUPLICATES,
        ),
        out_dir=args["out_dir"],
    )


def train_setup_2():
    from sentence_transformers.cross_encoder.losses import LambdaLoss

    args = read_data(data.parse_args(ARGS_DEFAULT), dataset_format='labeled-list')
    train_config = merge_configs(TRAIN_CONFIG_DEFAULT, args["train_config"])

    model = CrossEncoder(args["model"], device=training.device())
    loss = LambdaLoss(model, mini_batch_size=8)

    evaluator = CrossEncoderRerankingEvaluator(
        samples=args["ranking"],
        at_k=AT_K,
        name=EVALUATOR_NAME,
        # при обучении добавляем позитивы, чтобы оценивать только реранкер для выбора чекпоинта
        always_rerank_positives=True,
    )

    training.run_training(
        model=model,
        loss=loss,
        evaluator=evaluator,
        train_dataset=args["train_dataset"],
        train_config=train_config,
        trainer_cls=CrossEncoderTrainer,
        training_args_cls=CrossEncoderTrainingArguments,
        training_args_kwargs=dict(
            metric_for_best_model=f"{EVALUATOR_NAME}_ndcg@{AT_K}",
            batch_sampler=BatchSamplers.BATCH_SAMPLER,
        ),
        out_dir=args["out_dir"],
    )


if __name__ == "__main__":
    train_setup_1()
