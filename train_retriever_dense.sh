python train_retriever_dense.py \
--model "deepvk/USER-bge-m3" \
--train_config "configs/train/train_config.yaml" \
--data_dir "../data/dataset" \
--inv_qrels "inv_qrels.json" \
--passages_splits "passages_splits2.json" \
--passages "passages.json" \
--queries "queries.json" \
--train_dataset "hard_negatives/ds_h2.json" \
--out_dir "result"
