import argparse
import json
import logging
import math
import wandb
import os
from pathlib import Path
import sys

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    RandomRotation,
    RandomAffine,
    RandomVerticalFlip,
    RandomPerspective
)
from tqdm.auto import tqdm

import transformers
from torchvision.transforms.functional import to_pil_image
import numpy as np
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from evaluate import Metric, MetricInfo
from typing import Dict, List, Optional, Union

from utils import get_image_processor, SpecificityMetric, TimmConfig, TimmForImageClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.0.dev0")
require_version("datasets>=2.0.0", "To fix: pip install -r requirements.txt")

# Initialize logger
logger = get_logger(__name__)
        
# ---------------------------------------------
# Argument Parsing
# ---------------------------------------------        

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to True for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default="image",
        help="The name of the dataset column containing the image data. Defaults to 'image'.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=None,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help="The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset)."
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default="label",
        help="The name of the dataset column containing the labels. Defaults to 'label'."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different."
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use."
    )

    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )

    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )

    parser.add_argument(
        "--push_to_hub_model_id",
        type=str,
        default=None,
        help="El nombre del repositorio para sincronizar con el directorio local output_dir."
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help='The integration to report the results and logs to. Supported platforms are "tensorboard", "wandb", "comet_ml" and "clearml". Use "all" (default) to report to all integrations. Only applicable when --with_tracking is passed.'
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
 
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Si se especifica, ejecuta la evaluación del modelo."
    )

    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="The logging strategy to adopt during training."
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of update steps between two logs if logging_strategy='steps'"
    )

    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch", "no"],
        help="The evaluation strategy to adopt during training."
    )

    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch", "no"],
        help="The checkpoint save strategy to adopt during training."
    )

    parser.add_argument(
        "--load_best_model_at_end",
        type=str,
        default="false",
        help="Whether to load the best model at the end of training (true/false)"
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="If a value is passed, limit the total amount of checkpoints. Deletes the older checkpoints."
    )

    parser.add_argument(
        "--num_images_to_log",
        type=int,
        default=10,
        help="Number of images to log during evaluation."
    )

    # Parse arguments
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError(
                "Need an output_dir to create a repo when --push_to_hub or with_tracking is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args
    

# ---------------------------------------------
# Main Function
# ---------------------------------------------

def main():
    args = parse_args()
    
    # Set up the accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with=args.report_to if args.with_tracking else None,
                              project_dir=args.output_dir,
                              mixed_precision="bf16")

    # Send telemetry
    send_example_telemetry("cacao-hf3", args)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Configure the seed
    if args.seed:
        set_seed(args.seed)

    # Manage repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Infer repo_name
            repo_name = args.push_to_hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and get repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load the dataset
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, trust_remote_code=args.trust_remote_code)
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )

    # Check the columns of the dataset
    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["test"].column_names
    if args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {args.image_column_name} no encontrado en el dataset '{args.dataset_name}'. "
            "Asegúrate de establecer --image_column_name al nombre correcto de la columna de imagen."
        )
    if args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {args.label_column_name} no encontrado en el dataset '{args.dataset_name}'. "
            "Asegúrate de establecer --label_column_name al nombre correcto de la columna de etiquetas."
        )

    # Split the dataset if validation not exist
    """ args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"] """

    # Prepare the labels
    labels = dataset["train"].features[args.label_column_name].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and image processor
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
        trust_remote_code=args.trust_remote_code,
    )
    
    image_processor = get_image_processor(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as e:
        config = TimmConfig.from_pretrained(args.model_name_or_path)
        model = TimmForImageClassification.from_pretrained(args.model_name_or_path, config, num_labels=len(labels))

    # Preprocessing of the datasets
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(p=0.5),  
            RandomVerticalFlip(p=0.2),
            RandomRotation(degrees=15),  
            RandomAffine(degrees=0, translate=(0.1, 0.1)),  
            RandomPerspective(distortion_scale=0.2, p=0.5),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess(example_batch):
        """Apply training transformations in a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply validation transformations in a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        return example_batch


    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Establish training transformations
        train_dataset = dataset["train"].with_transform(preprocess)

        print(f"Original validation dataset size: {len(dataset['test'])}")
        if args.max_eval_samples is not None:
            dataset["test"] = dataset["test"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
            print(f"Validation dataset size after shuffling and selecting: {len(dataset['test'])}")

        # Establish validation transformations
        eval_dataset = dataset["test"].with_transform(preprocess_val)


    # Create DataLoaders
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    #body_params = [p for n,p in model.named_parameters() if "head" or "embed" not in n]
    #head_params = [p for n,p in model.named_parameters() if "head" or "embed" in n]
    #adamw_params = [p for p in body_params if p.ndim < 2] + head_params
    #muon_params = [p for p in body_params if p.ndim >= 2]
    #from muon import Muon
    #optimizer = Muon(muon_params, lr=args.learning_rate, momentum=0.95,
    #                adamw_params=adamw_params, adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.01)
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and calculation of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Preparing everything with the accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate training steps if necessary
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Recompute the number of epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Determine when to save accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Initialize trackers if is necessary
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard no puede registrar Enums, necesitamos el valor crudo
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("cacao-hf3", experiment_config)

    metric_accuracy = evaluate.load("accuracy")
    metric_specificity = SpecificityMetric()
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")

    #global metrics accuracy and specificity
    global_metrics = evaluate.combine([
                metric_accuracy,
                metric_specificity,
            ])

    per_class_metrics = evaluate.combine([
                metric_precision,
                metric_recall
            ])

    # Compute the batch size
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None  
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Reanudando desde el checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract epoch_{i} or step_{i}
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply gradient_accumulation_steps to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        print(f"Epoch {epoch}")
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first n batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_main_process:
                            image_processor.save_pretrained(args.output_dir)
                            api.upload_folder(
                                commit_message=f"Training in progress epoch {epoch}",
                                folder_path=args.output_dir,
                                repo_id=repo_id,
                                repo_type="model",
                                token=args.hub_token,
                            )

            if completed_steps >= args.max_train_steps:
                break

        if args.do_eval:
            model.eval()
            all_predictions = []
            all_references = []
            all_images = []

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                
                # Add batch to metrics
                per_class_metrics.add_batch(
                    predictions=predictions,
                    references=references
                )

                global_metrics.add_batch(
                    predictions=predictions,
                    references=references
                )

                metric_f1.add_batch(
                    predictions=predictions,
                    references=references
                )

                # Store for confusion matrix and image logging
                all_predictions.extend(predictions.cpu().numpy())
                all_references.extend(references.cpu().numpy())
                all_images.extend(batch["pixel_values"].cpu().numpy())
            
            metrics_result1 = per_class_metrics.compute(average="macro", zero_division=0)
            metrics_result2 = global_metrics.compute()
            metric_f1_result = metric_f1.compute(average="macro")

            if args.with_tracking and 'wandb' in args.report_to:
                current_lr = optimizer.param_groups[0]['lr']

                metrics_dict = {
                    "accuracy": metrics_result2["accuracy"],
                    "precision": metrics_result1["precision"],
                    "recall": metrics_result1["recall"],
                    "f1": metric_f1_result["f1"],
                    "specificity": metrics_result2["specificity"],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    "learning_rate": current_lr
                }

                # Log confusion matrix
                metrics_dict.update({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_references,
                        preds=all_predictions,
                        class_names=labels,
                    )
                })

                # Select a sample of images for log it
                num_images_to_log = min(args.num_images_to_log, len(all_images))
                sample_indices = np.random.choice(len(all_images), num_images_to_log, replace=False)
            
                wandb_images = []

                for idx in sample_indices:
                    img = all_images[idx]
                    pred = all_predictions[idx]
                    ref = all_references[idx]
                
                    # Denormalize image if needed
                    if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
                        img = img * np.array(image_processor.image_std).reshape(-1, 1, 1) + np.array(image_processor.image_mean).reshape(-1, 1, 1)
                        img = np.clip(img, 0, 1)

                    img_tensor = torch.tensor(img)
                    pil_img = to_pil_image(img_tensor)

                    caption = f"Real: {labels[ref]}, Prediction: {labels[pred]}"
                    wandb_images.append(wandb.Image(pil_img, caption=caption))
                    
                metrics_dict.update({"Predictions_vs_Real": wandb_images})

                accelerator.log(metrics_dict, step=completed_steps)
                
                
            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    image_processor.save_pretrained(args.output_dir)
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

    # End Training
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            metrics_dict = {
                "accuracy": metrics_result2["accuracy"],
                "precision": metrics_result1["precision"],
                "recall": metrics_result1["recall"],
                "f1": metrics_result1["f1"],
                "specificity": metrics_result2["specificity"],
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            }
            all_results = {f"eval_{k}": v for k, v in metrics_dict.items()}

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

if __name__ == "__main__":
    main()