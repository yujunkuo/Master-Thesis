## There are some tool functions for my research ##

# Import packages
import gc
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer, BertModel

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchviz import make_dot

from typing import List, Dict, Tuple, Optional


# Global Variables
MAX_CONTEXT_COUNT = 8  # 6 > 5


# Fix random seed
def set_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("**** The seed has been initialized ****")
    return None

    
# Load the dataset and group by Dialogue_ID
def load_df(file_path: str) -> pd.DataFrame:
    # Load Dataset
    df = pd.read_csv(file_path)
    # The shape of the Dataset
    print(f"The shape of the Dataset: {df.shape}")
    # How many unique dialogues
    print(f"The number of dialogues: {df.Dialogue_ID.nunique()}")
    # Dataset
    print(df.head())
    print("=" * 70)
    # Group by Dialogue_ID
    df = df.groupby("Dialogue_ID")
    return df


# # Count how many person-dialogues appear in the dialogue
# def count_person_dialogues(speakers: List[str]) -> int:
#     count, prev_speaker = 0, None
#     for speaker in speakers:
#         if speaker != prev_speaker:
#             count += 1
#         prev_speaker = speaker
#     return count


# # Print the statistical number of person-dialogues in each dataframe
# def print_num_of_person_dialogues_in_df(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
#     print("Number of person-dialogue in each dialogue:")
#     print("=" * 30)
#     for df in (train_df, valid_df, test_df):
#         res = []
#         for group_id in df.groups.keys():
#             # Get Speaker sequence
#             speakers = df.get_group(group_id).Speaker.values
#             person_dialogues = count_person_dialogues(speakers)
#             res.append(person_dialogues)
#         print(f"Max count: {max(res)}")
#         print(f"Min count: {min(res)}")
#         print(f"Avg count: {round(sum(res) / len(res), 2)}")
#         print("=" * 30)
#     return None


# Construct the DataFrame which feeds to the training data
def get_dialogues_df(df: pd.DataFrame, main_label_column: str, main_label_mapping: Dict, default_mapping: Optional[Dict]=None):
    # Construct new DataFrame
    dialogues_df = pd.DataFrame(columns=["context", "labels"])
    # Fill the new DataFrame
    for dialogue_id in tqdm(df.groups.keys()):
        # Get the dialogue with dialogue id
        dialogue = df.get_group(dialogue_id)
        # Get Speaker sequence
        speakers = dialogue.Speaker.values
        # Get the switch point of the speaker change
        switch_points = __get_switch_points_of_speakers(speakers)
        # Emotion Prediction in Conversation (EPC)
        for switch_point in switch_points:
            # Get context and labels
            context, labels = __get_context_and_labels(dialogue, main_label_column, switch_point)
            # Append the data into new dataframe
            if context != None:
                demo = dialogue.Utterance.values[switch_point]
                dialogues_df = dialogues_df.append({"context": context, "labels": labels, "demo": demo}, ignore_index=True)
    # Get the mapping dictionary of the labels (String to Number mapping)
    mapping = default_mapping if default_mapping else __get_str2num_labels_mapping(dialogues_df, main_label_mapping)
    # Map the labels to numbers
    dialogues_df.labels = dialogues_df.labels.map(lambda x: __map_labels(x, mapping=mapping))
    # The shape of the DataFrame
    print(f"The shape of the Dialogue DataFrame: {dialogues_df.shape}")
    print("=" * 50)
    # DataFrame
    dialogues_df = dialogues_df.reset_index(drop=True)
    return dialogues_df, mapping


    #             if group:
#                 grouped_utterances = __group_utterances_by_speakers(utterances[:switch_point], speakers[:switch_point])
#                 context = ["[SEP]".join(sub_group) for sub_group in grouped_utterances]
#             else:
#                 label = main_labels[switch_point]  # ERC: label = main_labels[switch_point-1]
#             # Padding all the length to 10
#             context_count, max_context_count = len(context), 10
#             if context_count < max_context_count:
#                 padding_length = max_context_count - context_count
#                 context.extend([""] * padding_length)
#                 label["sentiment_labels"].extend(["[SENTIMENT-PAD]"] * padding_length)
#                 label["DA_labels"].extend(["[DA-PAD]"] * padding_length)
#             else:
#                 context = context[-max_context_count:]
#                 label["sentiment_labels"] = label["sentiment_labels"][-max_context_count:]
#                 label["DA_labels"] = label["DA_labels"][-max_context_count:]
            # Make all data with context count of MAX_CONTEXT_COUNT


# Get the switch point of the speaker change
def __get_switch_points_of_speakers(speakers: List[str]):
    assert len(speakers) > 1
    switch_points = []
    for i in range(1, len(speakers)):
        if speakers[i] != speakers[i-1] and i >= MAX_CONTEXT_COUNT:
            switch_points.append(i)
#             # No overlapping
#             if (not switch_points) or i - switch_points[-1] >= MAX_CONTEXT_COUNT:
    # Generate one (multiple) data from one dialogue
#     return switch_points[-1:]
    return switch_points


# Get context and labels
def __get_context_and_labels(dialogue, main_label_column: str, switch_point: int):
    # Get Speaker sequence
    speakers = dialogue.Speaker.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    # Get Speakers mapping
    speakers_mapping = __get_speakers_mapping(speakers)
    if not speakers_mapping:
        return None, None
    # Get Context sequence
    utterances = dialogue.Utterance.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    context = __convert_utterances(utterances, speakers, speakers_mapping)
    # Get Main label (Target Label)
    main_label = dialogue[main_label_column].values[switch_point]
    # Get Future DA label
    future_DA_label = dialogue.DA.values[switch_point]
    # Get Sentiment label sequence
    sentiment_labels = dialogue.Sentiment.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    # Get DA label sequence
    DA_labels = dialogue.DA.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    # Get Big Five labels
    neuroticism = dialogue.Neuroticism.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    extraversion = dialogue.Extraversion.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    openness = dialogue.Openness.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    agreeableness = dialogue.Agreeableness.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    conscientiousness = dialogue.Conscientiousness.values[:switch_point][-MAX_CONTEXT_COUNT:].tolist()
    big_five_labels = __get_big_five_labels(speakers, speakers_mapping, neuroticism, extraversion, openness, agreeableness, conscientiousness)
    if not big_five_labels:
        return None, None
    # Get speaking order
    speaking_order = [speakers_mapping[speaker] for speaker in speakers]
    # Assertion check
    assert len(speakers) == len(utterances) == len(sentiment_labels) == len(DA_labels) == len(neuroticism)
    # Labels data
    labels = {
        "main_label": main_label,
        "future_DA_label": future_DA_label,
        "sentiment_labels": sentiment_labels,
        "DA_labels": DA_labels,
        "big_five_labels": big_five_labels,
        "speaking_order": speaking_order,
    }
    return context, labels


# # Group utterances by speakers
# def __group_utterances_by_speakers(utterances: List[str], speakers: List[str]) -> List[List[str]]:
#     assert len(utterances) == len(speakers)
#     result, temp = [], []
#     prev_speaker = None
#     for i in range(len(utterances)):
#         if speakers[i] != prev_speaker and temp:
#             result.append(temp[:])
#             temp = []
#         temp.append(utterances[i])
#         prev_speaker = speakers[i]
#     if temp:
#         result.append(temp[:])
#     return result


# Get Speakers mapping
def __get_speakers_mapping(speakers: List):
    # Convert speaker name into notation (e.g. A, B)
    notation_list = ("語者一", "語者二")
    unique_speakers = list(dict.fromkeys(speakers))
    if len(unique_speakers) != len(notation_list):
        return None
    speakers_mapping = {speaker: notation_list[idx] for idx, speaker in enumerate(unique_speakers)} # chr(ord("A") + prefix)
    return speakers_mapping


# Convert speaker name into notation and insert it into the utterances 
def __convert_utterances(utterances: List, speakers: List, speakers_mapping: Dict):
    # Prepend the speaker notation in front of each utterance (e.g. A: Hello everyone!)
    for i, (speaker, utterance) in enumerate(zip(speakers, utterances)):
        speaker_notation = speakers_mapping[speaker]
        new_utterance = speaker_notation + "：" + utterance
        utterances[i] = new_utterance
    return utterances


def __get_big_five_labels(speakers, speakers_mapping, neuroticism, extraversion, openness, agreeableness, conscientiousness):
    big_five_labels = dict()
    for i, speaker in enumerate(speakers):
        speaker_notation = speakers_mapping[speaker]
        big_five_label = (neuroticism[i], extraversion[i], openness[i], agreeableness[i], conscientiousness[i])
        if "unknown" in big_five_label:
            return None
        else:
            big_five_labels[speaker_notation] = big_five_label
    return big_five_labels


# Get the mapping dictionary of the labels (String to Number mapping)
def __get_str2num_labels_mapping(dialogues_df: pd.DataFrame, main_label_mapping: Dict):
    # Unique categories of each lable type
    main_label_set = {labels["main_label"] for labels in dialogues_df.labels.tolist()}
    sentiment_label_set = {label for labels in dialogues_df.labels.tolist() for label in labels["sentiment_labels"]}
    DA_label_set = {label for labels in dialogues_df.labels.tolist() for label in labels["DA_labels"]}
    # Mapping label string to number
    sentiment_label_mapping = {val: idx for idx, val in enumerate(sentiment_label_set)}
    DA_label_mapping = {val: idx for idx, val in enumerate(DA_label_set)}
    big_five_label_mapping = {"low": 0.0, "high": 1.0}
    speaking_order_mapping = {"語者一": 0, "語者二": 1}
    # Update final label mapping
    assert set(main_label_mapping) == main_label_set
    mapping = dict()
    mapping.update(sentiment_label_mapping)
    mapping.update(DA_label_mapping)
    mapping.update(big_five_label_mapping)
    mapping.update(speaking_order_mapping)
    mapping.update(main_label_mapping)
    # Print "How many labels"
    print(f"The number of main labels: {len(main_label_mapping)}")
    print(f"The number of sentiment labels: {len(sentiment_label_mapping)}")
    print(f"The number of DA labels: {len(DA_label_mapping)}")
    print(f"The number of big five labels: {len(big_five_label_mapping)}")
    print(f"The Mapping Dictionary: {mapping}")
    return mapping


# Map the labels to numbers
def __map_labels(labels, mapping):
    new_labels = {
        "main_label": mapping[labels["main_label"]], 
        "future_DA_label": mapping[labels["future_DA_label"]],
        "sentiment_labels": [mapping[label] for label in labels["sentiment_labels"]],
        "DA_labels": [mapping[label] for label in labels["DA_labels"]],
        "big_five_labels": {speaker: [mapping[x] for x in big_five] for speaker, big_five in labels["big_five_labels"].items()},
        "speaking_order": [mapping[label] for label in labels["speaking_order"]],
    }
    return new_labels


# Get the embedding of each utterance and save them
def get_dialogues_embedding(df: pd.DataFrame, tokenizer: BertTokenizer, bert: BertModel, device: torch.device, save_path: str=None):
    context_lists = df.context.values.tolist()
    embedding_tensor = torch.ones((len(context_lists), MAX_CONTEXT_COUNT, 768))
    for i in tqdm(range(len(context_lists))):
        for j, context in enumerate(context_lists[i]):
            bert_input = tokenizer(context, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
            attention_mask = bert_input['attention_mask'].to(device)
            input_id = bert_input['input_ids'].squeeze(1).to(device)
            final_inputs = {'input_ids': input_id, 'attention_mask': attention_mask}
            outputs = bert(**final_inputs)
            # # [Mean Pooling]
            # pooler_output = torch.mean(outputs.last_hidden_state, dim=1).detach().cpu()
            # [CLS]
            pooler_output = outputs.pooler_output.detach().cpu() # torch.Size([1, 768]) -> torch.Size([768])
            embedding_tensor[i].index_copy_(0, torch.tensor([j]), pooler_output)
        gc.collect()
        torch.cuda.empty_cache()
    # Save the BERT Embedding Tensor
    if save_path:
        torch.save(embedding_tensor, save_path)
    return embedding_tensor


# Get the weighted loss
def multi_task_loss(main_output, future_DA_output, sentiment_output, DA_output, speaker1_output, speaker2_output, main_labels, future_DA_labels, sentiment_labels, DA_labels, speaker1_labels, speaker2_labels, a=1.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0):
    # Class Weights (1436, 2862, 826) -> (1.0, 0.50, 1.74)
    # class_weights = torch.FloatTensor([1.0, 0.50, 1.74]).cuda()
    # main_loss = F.cross_entropy(main_output, main_labels, weight=class_weights)
#     DA_class_weights = torch.FloatTensor([ 12.8421,   8.6069,   3.7169,  20.0592,   0.2625,   6.8371, 105.5286,
#          10.9331,  15.7608,   7.3107,   9.9883,   1.6192,   7.2669,   3.4923,
#           6.1447,   0.1256,   0.2161,  26.3822,   3.3757]).cuda()
    # Calculate each task's loss
    main_loss = F.cross_entropy(main_output, main_labels)
    future_DA_loss = F.cross_entropy(future_DA_output, future_DA_labels)
    sentiment_loss = F.cross_entropy(sentiment_output, sentiment_labels)
    DA_loss = F.cross_entropy(DA_output, DA_labels)
    speaker1_loss = F.binary_cross_entropy_with_logits(speaker1_output, speaker1_labels)
    speaker2_loss = F.binary_cross_entropy_with_logits(speaker2_output, speaker2_labels)
    
#     # test1
#     future_DA_loss = 0.5 * future_DA_loss
#     DA_loss = 0.5 * DA_loss
    
    # Calculate total loss with weights for each task
    total_loss = a * main_loss + b * future_DA_loss + c * sentiment_loss + d * DA_loss + e * speaker1_loss + f * speaker2_loss
#     weight = F.softmax(torch.randn(3), dim=-1) # RLW
#     weight = 0.05 * weight
#     total_loss = 0.95 * main_loss + weight[0] * future_DA_loss + weight[1] * sentiment_loss + weight[2] * DA_loss
#     total_loss = a * main_loss + b * future_DA_loss
    loss_dict = {
        "total_loss": total_loss,
        "main_loss": main_loss,
        "future_DA_loss": future_DA_loss,
        "sentiment_loss": sentiment_loss,
        "DA_loss": DA_loss,
        "speaker1_loss": speaker1_loss,
        "speaker2_loss": speaker2_loss,
    }
    return loss_dict


# # Plot the result
# def plot_result(train_list: List[float], valid_list: List[float], label: str) -> None:
#     plt.plot(range(1, len(train_list)+1), train_list, label=f"train_{label}")
#     plt.plot(range(1, len(valid_list)+1), valid_list, label=f"valid_{label}")
#     plt.legend()
#     plt.xlabel("Epochs")
#     plt.ylabel(label.title())
#     plt.show()
#     return None


# Plot the result
def plot_result(train_list: List[List[float]], valid_list: List[List[float]], labels: List[str]) -> None:
    if len(train_list) != len(valid_list) or len(train_list) != len(labels):
        raise ValueError("The lengths of train_list, valid_list, and labels must be the same.")

    if len(train_list) == 1:
        plt.plot(range(1, len(train_list[0])+1), train_list[0], label=f"train_{labels[0]}")
        plt.plot(range(1, len(valid_list[0])+1), valid_list[0], label=f"valid_{labels[0]}")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(labels[0].title())
        plt.show()
    else:
        num_plots = len(train_list) if "total_loss" not in labels else len(train_list) - 1
        num_rows = num_plots // 3 + (num_plots % 3 > 0)
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4*num_rows))
        axes = axes.flatten()
        
        i = 0
        for ax in axes[:num_plots]:
            if labels[i] == "total_loss":
                i += 1
            ax.plot(range(1, len(train_list[i])+1), train_list[i], label=f"train_{labels[i]}")
            ax.plot(range(1, len(valid_list[i])+1), valid_list[i], label=f"valid_{labels[i]}")
            ax.legend()
            ax.set_xlabel("Epochs")
            ax.set_ylabel(labels[i].title())
            i += 1

        for j in range(num_plots, num_rows*3):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


# Plot the report
def plot_report(test_true: List[int], test_pred: List[int], display_labels: List[str]) -> None:
    print(metrics.classification_report(test_true, test_pred, target_names=display_labels, digits=4))
    return None


# Plot Confusion Matrix
def plot_confusion_matrix(test_true: List[int], test_pred: List[int], labels: List[int], display_labels: List[str]) -> None:
    cm = confusion_matrix(test_true, test_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.show()
    return None


# Plot Model Architecture
def plot_model_architecture(model, device: torch.device):
    sample_embeddings = torch.randn(1, MAX_CONTEXT_COUNT, 768)
    sample_speaking_order = torch.tensor([[0] * (MAX_CONTEXT_COUNT // 2) + [1] * (MAX_CONTEXT_COUNT - (MAX_CONTEXT_COUNT // 2))])
    sample_embeddings, sample_speaking_order = sample_embeddings.to(device), sample_speaking_order.to(device)
    return make_dot(model(sample_embeddings, sample_speaking_order), params=dict(model.named_parameters()))
