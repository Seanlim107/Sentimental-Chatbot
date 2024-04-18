import torch
from torch.utils.data import Dataset, DataLoader

class DialogueDataset(Dataset):
    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx]

def collate_fn(batch):
    max_num_sentences = max(len(dialogue) for dialogue in batch)
    max_sentence_len = max(len(sentence) for dialogue in batch for sentence in dialogue)
    
    padded_batch = []
    sentence_masks = []
    dialogue_masks = []
    for dialogue in batch:
        padded_dialogue = []
        sentence_mask = []
        for sentence in dialogue:
            padded_sentence = sentence + [0] * (max_sentence_len - len(sentence))
            padded_dialogue.append(padded_sentence)
            sentence_mask.append(1)
        # Pad dialogue to ensure all dialogues have the same number of sentences
        padded_dialogue += [[0] * max_sentence_len] * (max_num_sentences - len(dialogue))
        padded_batch.append(padded_dialogue)
        sentence_masks.append(sentence_mask)
    
    # Pad dialogues to ensure all batches have the same number of dialogues
    padded_batch += [[[0] * max_sentence_len] * max_num_sentences] * (max_num_sentences - len(batch))
    sentence_masks += [[0] * max_sentence_len] * (max_num_sentences - len(batch))
    
    dialogue_masks = [[1] * len(dialogue) + [0] * (max_num_sentences - len(dialogue)) for dialogue in batch]
    # print(sentence_masks)
    
    return torch.tensor(padded_batch), torch.tensor(sentence_masks), torch.tensor(dialogue_masks)


# Example dialogues
dialogues = [
    [[1, 2, 3], [4, 5], [6]],
    [[7, 8], [9, 10, 11]],
    [[12]]
]

# Create dataset and dataloader
dataset = DialogueDataset(dialogues)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Iterate through batches
for batch_dialogues, batch_sentence_masks, batch_dialogue_masks in dataloader:
    print("Batch Dialogues:")
    print(batch_dialogues)
    print("Batch Sentence Masks:")
    print(batch_sentence_masks)
    print("Batch Dialogue Masks:")
    print(batch_dialogue_masks)
    print()
