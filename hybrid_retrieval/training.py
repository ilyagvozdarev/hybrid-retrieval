from pathlib import Path
from datetime import datetime

import torch

from hybrid_retrieval.util.io import write_json
from hybrid_retrieval.util.param_estimate_meminfo import format_report, stats_from_model
from hybrid_retrieval.util.loss_inspect import describe_loss
from hybrid_retrieval.util.collect import collect


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


def make_training_args(train_config, training_args_cls, output_dir, run_name, extra):
    """
    train_config has already been merged from the script defaults and the yaml;
    fp16/bf16 are added according to the hardware.
    extra — TrainingArguments fields specific to a given script; they override the common ones.
    """
    is_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    train_config["optim"] = {
        "fp16": not is_bf16,
        "bf16": is_bf16,
        **train_config["optim"],
    }

    kwargs = dict(
        output_dir=output_dir,
        **train_config["training"],
        **train_config["optim"],
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=0.05,
        run_name=run_name,
        seed=train_config["seed"],
    )
    return training_args_cls(**{**kwargs, **extra})


def run_training(
    *,
    model,
    loss,
    evaluator,
    train_dataset,
    train_config,
    trainer_cls,
    training_args_cls,
    training_args_kwargs,
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
        extra=training_args_kwargs,
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
        dict(**train_config, loss=describe_loss(loss), batch_sampler=targs.batch_sampler.value),
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
