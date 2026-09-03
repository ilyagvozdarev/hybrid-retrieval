from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.base.sampler import BatchSamplers

from ir_pipeline import data, training
from ir_pipeline.config import merge_configs


ARGS_DEFAULT = dict(**data.ARGS_DEFAULT, model="deepvk/USER-bge-m3")

TRAIN_CONFIG_DEFAULT = dict(
    optim=dict(
        gradient_checkpointing=True,
        optim="adamw_torch_fused"
    ),
    training=dict(
        num_train_epochs=2,
        train_batch_size=16,
        learning_rate=2.0e-5,
        warmup_steps=0.1
    ),
    seed=42
)

EVALUATOR_NAME = "ir_evaluator"


def main():
    args = data.read_data(data.parse_args(ARGS_DEFAULT))

    model = SentenceTransformer(args["model"], device=training.device())

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)

    evaluator = InformationRetrievalEvaluator(
        queries=args["queries_eval"],
        corpus=args["passages"],
        relevant_docs=args["qrels_eval"],
        name=EVALUATOR_NAME,
        map_at_k=[20],
        show_progress_bar=True,
        write_predictions=True,
    )

    training.run_training(
        model=model,
        loss=loss,
        evaluator=evaluator,
        train_dataset=args["train_dataset"],
        train_config=merge_configs(TRAIN_CONFIG_DEFAULT, args["train_config"]),
        # evaluator даёт метрики вида {name}_{similarity_fn_name}_{metric}
        metric_for_best_model=f"{EVALUATOR_NAME}_cosine_ndcg@10",
        trainer_cls=SentenceTransformerTrainer,
        training_args_cls=SentenceTransformerTrainingArguments,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        out_dir=args["out_dir"],
    )


if __name__ == "__main__":
    main()
