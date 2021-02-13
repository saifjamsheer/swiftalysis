import gpt_2_simple as gpt2
import os
import requests

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name) 

LYRICS_PATH = 'datasets/inputs.txt'

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              LYRICS_PATH,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps

gpt2.generate(sess)