from datasets import load_dataset

# load cifar10 (only small portion for demonstration purposes) 
train_ds, test_ds = load_dataset('cifar10', split=['train[:500]', 'test[:200]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {idx:label for idx,label in enumerate(train_ds.features['label'].names)}
label2id = {label:idx for idx, label in id2label.items()}


from transformers import PerceiverFeatureExtractor

feature_extractor = PerceiverFeatureExtractor()

import numpy as np

def preprocess_images(examples):
    examples['pixel_values'] = feature_extractor(examples['img'], return_tensors="pt").pixel_values
    return examples

# Set the transforms
train_ds.set_transform(preprocess_images)
val_ds.set_transform(preprocess_images)
test_ds.set_transform(preprocess_images)


from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_batch_size = 2
eval_batch_size = 2

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

from transformers import PerceiverForImageClassificationLearned 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned",
                                                               num_labels=10,
                                                               id2label=id2label,
                                                               label2id=label2id,
                                                               ignore_mismatched_sizes=True)
model.to(device)


from transformers import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(2):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for batch in train_dataloader:
        # get the inputs; 
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # evaluate
        predictions = outputs.logits.argmax(-1).cpu().detach().numpy()
        accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
        print(f"Loss: {loss.item()}, Accuracy: {accuracy}")


from tqdm.notebook import tqdm
from datasets import load_metric

accuracy = load_metric("accuracy")

model.eval()
for batch in val_dataloader:
    # get the inputs; 
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # forward pass
    outputs = model(inputs=inputs, labels=labels)
    logits = outputs.logits 
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch["labels"].numpy()
    accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)