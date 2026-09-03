"""Запуск обучения: всё, что одинаково для dense / splade / реранкера."""

from pathlib import Path
from datetime import datetime

import torch
from sentence_transformers.base.sampler import BatchSamplers

from ir_pipeline.util.io import write_json
from ir_pipeline.util.param_estimate_meminfo import format_report, stats_from_model
from ir_pipeline.util.loss_inspect import describe_loss
from ir_pipeline.util.collect import collect


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_info(model):
    print(f"model.config: {model.config}\n")
    print(model[0].auto_model)
    print(f"attn_implementation: {model[0].auto_model.config._attn_implementation}\n")
    print(f"model.prompts: {model.prompts}")

    stats = stats_from_model(model)
    print(format_report(stats, optimizer="adamw", amp=None, allocator_aware=True))
    print(f"max_seq_length = {model.max_seq_length}")


def make_run_name(model):
    model_name = model[0].auto_model.config._name_or_path
    return f"{model_name.split('/')[-1]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def make_training_args(train_config, training_args_cls, output_dir, run_name,
                       metric_for_best_model, batch_sampler):
    """train_config уже слит из дефолтов скрипта и yaml; fp16/bf16 добавляются по железу."""
    is_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    train_config["optim"] = {
        "fp16": not is_bf16,
        "bf16": is_bf16,
        **train_config["optim"],       # явное значение из конфига важнее автоопределения
    }

    return training_args_cls(
        output_dir=output_dir,
        **train_config["training"],
        **train_config["optim"],
        batch_sampler=batch_sampler,
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        logging_steps=0.05,
        run_name=run_name,
        seed=train_config["seed"],
    )


def run_training(
    *,
    model,
    loss,
    evaluator,
    train_dataset,
    train_config,
    metric_for_best_model,
    trainer_cls,
    training_args_cls,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    callbacks=(),
    out_dir="result",
):
    model_info(model)

    run_name = make_run_name(model)
    output_dir = Path(out_dir) / run_name

    targs = make_training_args(
        train_config=train_config,
        training_args_cls=training_args_cls,
        output_dir=output_dir,
        run_name=run_name,
        metric_for_best_model=metric_for_best_model,
        batch_sampler=batch_sampler,
    )

    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=list(callbacks),
    )

    write_json(
        dict(**train_config, loss=describe_loss(loss), batch_sampler=batch_sampler.value),
        output_dir / "settings.json",
    )

    try:
        trainer.train()
    finally:
        trainer.model = None
        del trainer
        model.zero_grad(set_to_none=True)
        collect()

    return output_dir
