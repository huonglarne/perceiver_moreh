from datasets import load_dataset

train_ds, test_ds = load_dataset("imdb", split=['train[:10]+train[-10:]', 'test[:5]+test[-5:]'])
labels = train_ds.features['label'].names

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


from transformers import PerceiverTokenizer

tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")

train_ds = train_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True),
                        batched=True)
test_ds = test_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True),
                      batched=True)


train_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=1)

from transformers import PerceiverForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PerceiverForSequenceClassification.from_pretrained("deepmind/language-perceiver",
                                                               num_labels=2,
                                                               id2label=id2label,
                                                               label2id=label2id)
model.to(device)


from transformers import AdamW
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(20):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for batch in train_dataloader:
         # get the inputs; 
         inputs = batch["input_ids"].to(device)
         attention_mask = batch["attention_mask"].to(device)
         labels = batch["label"].to(device)

         # zero the parameter gradients
         optimizer.zero_grad()

         # forward + backward + optimize
         outputs = model(inputs=inputs, attention_mask=attention_mask, labels=labels)
         loss = outputs.loss
         loss.backward()
         optimizer.step()

         # evaluate
         predictions = outputs.logits.argmax(-1).cpu().detach().numpy()
         accuracy = accuracy_score(y_true=batch["label"].numpy(), y_pred=predictions)
         print(f"Loss: {loss.item()}, Accuracy: {accuracy}")


from tqdm.notebook import tqdm
from datasets import load_metric

accuracy = load_metric("accuracy")

model.eval()
for batch in test_dataloader:
      # get the inputs; 
      inputs = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["label"].to(device)

      # forward pass
      outputs = model(inputs=inputs, attention_mask=attention_mask)
      logits = outputs.logits 
      predictions = logits.argmax(-1).cpu().detach().numpy()
      references = batch["label"].numpy()
      accuracy.add_batch(predictions=predictions, references=references)

final_score = accuracy.compute()
print("Accuracy on test set:", final_score)


text = "I loved this movie, it's super good."

input_ids = tokenizer(text, return_tensors="pt").input_ids

# forward pass
outputs = model(inputs=input_ids.to(device))
logits = outputs.logits 
predicted_class_idx = logits.argmax(-1).item()

print("Predicted:", model.config.id2label[predicted_class_idx])