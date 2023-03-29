import base64
import functools
import os
import pickle
import ssl
import re
import subprocess
import tempfile

from urllib import request

import cv2
import imageio
import numpy as np
import scipy.io.wavfile

from IPython.display import HTML


# Utilities to fetch videos from UCF101 dataset
UCF_ROOT = 'https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/'
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()

def list_ucf_videos():
  """Lists videos available in UCF101 dataset."""
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode('utf-8')
    videos = re.findall('(v_[\w_]+\.avi)', index)
    _VIDEO_LIST = sorted(set(videos))
  return list(_VIDEO_LIST)

def fetch_ucf_video(video):
  """Fetchs a video and cache into local filesystem."""
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)
    print('Fetching %s => %s' % (urlpath, cache_path))
    data = request.urlopen(urlpath, context=unverified_context).read()
    open(cache_path, "wb").write(data)
  return cache_path

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
  with open('./animation.gif', 'rb') as f:
    gif_64 = base64.b64encode(f.read()).decode('utf-8')
  return HTML('<img src="data:image/gif;base64,%s"/>' % gif_64)

def play_audio(data, sample_rate=48000):
  scipy.io.wavfile.write('tmp_audio.wav', sample_rate, data)

  with open('./tmp_audio.wav', 'rb') as f:
    audio_64 = base64.b64encode(f.read()).decode('utf-8')
  return HTML('<audio controls src="data:audio/wav;base64,%s"/>' % audio_64)

def table(elements):
  row = ['<td>%s</td>' % el.data for el in elements]
  return HTML('<table><tr>%s</tr></table>' % ''.join(row))


video_names = list_ucf_videos()
video_path = fetch_ucf_video(video_names[0])

# Extract audio using FFMPEG and encode as pcm float wavfile (only format readable by scipy.io.wavfile).
command = f"""yes | ffmpeg -i {video_path}  -c copy  -f wav -map 0:a pcm_f32le -ar 48000 output.wav"""
output = subprocess.getoutput(command)
# print(output)

sample_rate, audio = scipy.io.wavfile.read("output.wav")
if audio.dtype == np.int16:
  audio = audio.astype(np.float32) / 2**15
elif audio.dtype != np.float32:
  raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')

video = load_video(video_path)


from transformers import PerceiverForMultimodalAutoencoding
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver", 
                                                                   low_cpu_mem_usage=True)
model.to(device)

def autoencode_video(images, audio):
  
  # only create entire video once as inputs
  inputs = {'image': torch.from_numpy(np.moveaxis(images, -1, 2)).float().to(device),
          'audio': torch.from_numpy(audio).to(device),
          'label': torch.zeros((images.shape[0], 700)).to(device)}
  
  nchunks = 128
  reconstruction = {}
  for chunk_idx in range(nchunks):
        image_chunk_size = np.prod(images.shape[1:-1]) // nchunks
        audio_chunk_size = audio.shape[1] // SAMPLES_PER_PATCH // nchunks
        subsampling = {
            'image': torch.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            'audio': torch.arange(
                audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            'label': None,
        }
        
        # forward pass
        with torch.no_grad():
          outputs = model(inputs=inputs, subsampled_output_points=subsampling)

        output = {k:v.cpu() for k,v in outputs.logits.items()}
        
        reconstruction['label'] = output['label']
        if 'image' not in reconstruction:
          reconstruction['image'] = output['image']
          reconstruction['audio'] = output['audio']
        else:
          reconstruction['image'] = torch.cat(
              [reconstruction['image'], output['image']], dim=1)
          reconstruction['audio'] = torch.cat(
              [reconstruction['audio'], output['audio']], dim=1)
          
        del outputs
        
  # finally, reshape image and audio modalities back to original shape
  reconstruction['image'] = torch.reshape(reconstruction['image'], images.shape)
  reconstruction['audio'] = torch.reshape(reconstruction['audio'], audio.shape)
  return reconstruction

  return None

AUDIO_SAMPLES_PER_FRAME = 48000 // 25
SAMPLES_PER_PATCH = 16

# Auto-encode the first 16 frames of the video and one of the audio channels
reconstruction = autoencode_video(video[None, :16], audio[None, :16*AUDIO_SAMPLES_PER_FRAME, 0:1])

# Print top 5 predicted labels
scores, indices = torch.topk(torch.softmax(reconstruction["label"], dim=1), k=5)
for score, index in zip(scores[0], indices[0]):
  print("%s: %s" % (model.config.id2label[index.item()], score.item()))