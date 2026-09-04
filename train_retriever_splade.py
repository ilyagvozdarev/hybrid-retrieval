'''
Running the Script:

```bash
python train_retriever_splade.py \
--model "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1" \
--train_config "configs/train/train_config.yaml" \
--data_dir "./data/dataset" \
--inv_qrels "inv_qrels.json" \
--passages_splits "passages_splits2.json" \
--passages "passages.json" \
--queries "queries.json" \
--train_dataset "hard_negatives/ds_h2.json" \
--out_dir "result"
```
'''

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, CachedSpladeLoss
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.modules import SpladePooling, MLMTransformer
from sentence_transformers.sparse_encoder.callbacks import SpladeRegularizerWeightSchedulerCallback
from sentence_transformers.base.sampler import BatchSamplers

from hybrid_retrieval import data, training
from hybrid_retrieval.config import merge_configs


ARGS_DEFAULT = dict(
    **data.ARGS_DEFAULT,
    model = dict(default="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
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

EVALUATOR_NAME = "irs_evaluator"
MAX_SEQ_LENGTH = 512
CHUNK_SIZE = 128


def build_model(model_name):
    model = SparseEncoder(
        modules=[
            MLMTransformer(model_name, max_seq_length=MAX_SEQ_LENGTH),
            SpladePooling(pooling_strategy="max", chunk_size=CHUNK_SIZE),
        ],
        device=training.device(),
        similarity_fn_name="dot",
    )
    model[-1].chunk_size = CHUNK_SIZE
    return model


def main():
    args = data.read_data(data.parse_args(ARGS_DEFAULT))

    model = build_model(args["model"])

    loss = CachedSpladeLoss(
        mini_batch_size=8,
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=3e-3,
        # use_document_regularizer_only=True,
        document_regularizer_weight=3e-3,
    )

    evaluator = SparseInformationRetrievalEvaluator(
        queries=args["queries_eval"],
        corpus=args["passages"],
        relevant_docs=args["qrels_eval"],
        name=EVALUATOR_NAME,
        map_at_k=[20, 100],
        show_progress_bar=True,
        write_predictions=True,
    )

    training.run_training(
        model=model,
        loss=loss,
        evaluator=evaluator,
        train_dataset=args["train_dataset"],
        train_config=merge_configs(TRAIN_CONFIG_DEFAULT, args["train_config"]),
        trainer_cls=SparseEncoderTrainer,
        training_args_cls=SparseEncoderTrainingArguments,
        training_args_kwargs=dict(
            metric_for_best_model=f"{EVALUATOR_NAME}_dot_ndcg@10",
            batch_sampler=BatchSamplers.NO_DUPLICATES,
        ),
        callbacks=[SpladeRegularizerWeightSchedulerCallback(loss=loss, warmup_ratio=1 / 3)],
        out_dir=args["out_dir"],
    )


if __name__ == "__main__":
    main()
