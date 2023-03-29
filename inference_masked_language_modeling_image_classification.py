from transformers import PerceiverTokenizer, PerceiverForMaskedLM

tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")
model = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver") 

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

text = "This is an incomplete sentence where some words are missing."
# prepare input
encoding = tokenizer(text, padding="max_length", return_tensors="pt")
# mask " missing.". 
encoding.input_ids[0, 52:61] = tokenizer.mask_token_id
inputs, input_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)

# forward pass
outputs = model(inputs=inputs, attention_mask=input_mask)
logits = outputs.logits
masked_tokens_predictions = logits[0, 51:61].argmax(dim=-1)
print(tokenizer.decode(masked_tokens_predictions))

from PIL import Image
import requests

url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


from transformers import PerceiverFeatureExtractor, PerceiverForImageClassificationLearned

del model
feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-learned")
model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")

model.to(device)

# prepare input
encoding = feature_extractor(image, return_tensors="pt")
inputs, input_mask = encoding.pixel_values.to(device), None
# forward pass
outputs = model(inputs, input_mask)
logits = outputs.logits

print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])


from transformers import PerceiverForImageClassificationFourier

del model
model = PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier")
model.to(device)

# prepare input
encoding = feature_extractor(image, return_tensors="pt")
inputs, input_mask = encoding.pixel_values.to(device), None
# forward pass
outputs = model(inputs, input_mask)
logits = outputs.logits

print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])
