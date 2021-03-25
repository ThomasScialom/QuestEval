from typing import Tuple, Optional
import logging
import os
import csv
import random
from tqdm.auto import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler

from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import get_constant_schedule
from transformers.trainer_utils import PredictionOutput

from safeval.utils import (
    T2TDataCollator,
    SummarizationDataset,
    ModelArguments, DataTrainingArguments
)

logger = logging.getLogger(__name__)


class CustomRandomSampler(RandomSampler):
    def __init__(self, *args, **kwargs):
        self.pool_size = kwargs.pop("pool_size")
        self.seed = kwargs.pop("seed")

        super(CustomRandomSampler, self).__init__(*args, **kwargs)

        if self.pool_size > len(self.data_source):
            self.custom_iter_function = super(CustomRandomSampler, self).__iter__

        else:
            max_pool_index = len(self.data_source) // self.pool_size
            indexs = [
                list(range(pool_index * self.pool_size, (pool_index + 1) * self.pool_size))
                for pool_index in range(max_pool_index)
            ]
            if len(self.data_source) % self.pool_size > 0:
                indexs.append(
                    list(range(max_pool_index * self.pool_size, len(self.data_source)))
                )

            random.seed(self.seed)
            indexs = [random.sample(pool, len(pool)) for pool in indexs]
            indexs = [i for pool in indexs for i in pool]

            self.custom_iter_function = lambda: iter(indexs)

    def __iter__(self):
        return self.custom_iter_function()


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.eval_output = kwargs.pop("eval_output", False)
        self.eval_output_name = kwargs.pop("eval_output_name", "")
        self.pool_size = kwargs.pop("pool_size", None)
        self.is_eval_output_required = False
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            CustomRandomSampler(self.train_dataset, pool_size=self.pool_size, seed=self.args.seed)
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        # Able to use constant scheduler at warmup steps == 0
        optimizer, scheduler = super(CustomTrainer, self).get_optimizers(num_training_steps)

        if self.args.warmup_steps == 0:
            scheduler = get_constant_schedule(optimizer)

        return optimizer, scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
    ) -> PredictionOutput:
        """
        Overwriting HF method :
        -> to allow different sizes of output
        -> to store output by batch
        -> to make it simple

        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            # WARNING: not sure working in multi gpu !
            with torch.no_grad():
                source_ids, source_mask = inputs["input_ids"], inputs["attention_mask"]
                generated_ids = self.model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    use_cache=True,
                    decoder_start_token_id=None,
                )
            batch_preds = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            if self.eval_output is True and self.is_eval_output_required is True:
                # Storing one output batch
                filename = "eval_output" + (f"_{self.eval_output_name}" if self.eval_output_name else "") + ".tsv"
                output_path = os.path.join(self.args.output_dir, filename)
                with open(output_path, 'a') as f:
                    writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    for item in batch_preds:
                        writer.writerow([item])

        metrics = {}

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO: in future remove slurm part
    if data_args.from_slurm == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    filename = "eval_output" + (f"_{data_args.eval_output_name}" if data_args.eval_output_name else "") + ".tsv"
    if (
        os.path.exists(training_args.output_dir)
        and data_args.eval_output is True
        and os.path.exists(os.path.join(training_args.output_dir, filename))
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output eval file already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    if (
        os.path.exists(training_args.output_dir)
        and data_args.eval_output is True
        and os.path.exists(os.path.join(training_args.output_dir, filename))
        and training_args.overwrite_output_dir is True
    ):
        os.remove(os.path.join(training_args.output_dir, filename))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer_cls = T5Tokenizer
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.model_name_or_path,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
    )

    model.resize_token_embeddings(len(tokenizer))

    # Get datasets
    logger.info('loading dataset')

    train_dataset = None
    pool_size = 400000
    if training_args.do_train:
        train_dataset = SummarizationDataset(
            tokenizer,
            data_dir=data_args.data_dir,
            type_path="train",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            pool_size=pool_size,
        )

    valid_dataset = SummarizationDataset(
        tokenizer,
        data_dir=data_args.data_dir,
        type_path="dev",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        pool_size=pool_size,
    )

    logger.info('finished loading dataset')

    # Initialize our Trainer
    training_args.save_steps = 0
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(
            tokenizer=tokenizer,
            mode="training",
        ),
        eval_output=data_args.eval_output,
        eval_output_name=data_args.eval_output_name,
        pool_size=pool_size
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        trainer.is_eval_output_required = True
        eval_output = trainer.evaluate()
        trainer.is_eval_output_required = False

        filename = "eval_results" + (f"_{data_args.eval_output_name}" if data_args.eval_output_name else "") + ".txt"
        output_eval_file = os.path.join(training_args.output_dir, filename)
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results


if __name__ == "__main__":
    main()
