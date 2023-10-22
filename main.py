import os
import re
import pandas as pd
from datasets import Dataset
import numpy as np
import random
import torch
import fire
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertTokenizer
)


from transformers import BartForConditionalGeneration
from statistics import mean

def main(
        file_path:str = "E:/COMPSCI 760/dataset"
):
    def load_data_from_file():
        all_files = os.listdir(file_path)
        all_df = pd.DataFrame()

        for file in all_files:
            inputFile = open(file_path+'/'+file, encoding='utf8').readlines()
            data = ''.join(inputFile)

            parts = re.split("本院认为，|本院意见，",data, maxsplit=1)

            data_dict = {}

            if len(parts) == 2:
                data_dict["source"] = parts[0]
                data_dict["target"] = parts[1]
            else:
                data_dict["source"] = data
                data_dict["target"] = ""

            data_df = pd.DataFrame(data_dict, index=[0])
            all_df = pd.concat([all_df, data_df], ignore_index=True)
        all_df = all_df[all_df['target'].str.strip() != '']
        new_dataset = Dataset.from_pandas(all_df)
        train_index = int(0.8 * len(new_dataset))
        val_index = int(0.9 * len(new_dataset))

        train_dataset = new_dataset[:train_index]
        val_dataset = new_dataset[train_index:val_index]
        test_dataset = new_dataset[val_index:]

        return train_dataset, val_dataset, test_dataset

    train_df_original, val_df_original, test_df_original = load_data_from_file()
    train_df_original = Dataset.from_dict(train_df_original)
    val_df_original = Dataset.from_dict(val_df_original)
    test_df_original = Dataset.from_dict(test_df_original)
    rouge = load_metric("rouge")

    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    encoder_max_length = 512
    decoder_max_length = 512
    batch_size = 1
    learning_rate = 3e-5
    weight_decay = 0.01
    num_train_epochs = 100
    random_seed = 3407

    def set_seed(seed: int = 3407):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")
    set_seed(random_seed)

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["source"],
            padding="max_length",
            truncation=True,
            max_length=encoder_max_length,
        )
        outputs = tokenizer(
            batch["target"],
            padding="max_length",
            truncation=True,
            max_length=decoder_max_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # since above lists are references, the following line changes the 0 index for all samples
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return  batch



    # map train data
    train_df = train_df_original.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["source", "target"]
    )

    # map val data
    val_df = val_df_original.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["source", "target"]
    )
    test_df = val_df_original.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["source", "target"]
    )


    # set Python list to PyTorch tensor
    train_df.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    val_df.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    test_df.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )


    # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        fp16=False,
        # fp16_backend='apex',
        output_dir="./",
        lr_scheduler_type="cosine",
        save_total_limit=2,
        gradient_accumulation_steps=1,
        optim="adafactor",
        load_best_model_at_end=True,
        group_by_length=True,
        gradient_checkpointing=True,
        seed=3407
    )

    # compute Rouge score during validation

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge2_output = rouge.compute(predictions=pred_str,references=labels_str,rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge2_output.precision, 4),
            "rouge2_recall": round(rouge2_output.recall, 4),
            "rouge2_fmeasure": round(rouge2_output.fmeasure, 4)
        }

    # load model + enable gradient checkpointing & disable cache for checkpointing
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese",use_cache=False)

    model.config.num_beams = 4
    model.config.max_length = 512
    model.config.min_length = 256
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_df,
        eval_dataset=val_df,
    )

    # start training
    # torch.autograd.set_detect_anomaly(True)
    trainer.train()

    predictions = trainer.predict(test_df)
    metrics = compute_metrics(predictions)
    print("ROUGE-2 Precision:", metrics['rouge2_precision'])
    print("ROUGE-2 Recall:", metrics['rouge2_recall'])
    print("ROUGE-2 Fmeasure:",metrics['rouge2_fmeasure'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    def generate_answer(batch):
        inputs_dict = tokenizer(batch["source"], padding="max_length", max_length=512, return_tensors="pt", truncation=True)

        # Move tensors to the device
        input_ids = inputs_dict.input_ids.to(device)

        predicted_abstract_ids = model.generate(input_ids)
        batch["predicted_target"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
        return batch

    result = test_df_original.map(generate_answer, batch_size=1, batched=True)
    result_df = pd.DataFrame(result)
    result_df.to_csv("result.csv")


if __name__ == '__main__':
    fire.Fire(main)













