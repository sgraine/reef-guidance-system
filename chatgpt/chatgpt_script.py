import os
from openai import OpenAI
import numpy as np
import base64, io
import tiktoken
import json
from PIL import Image

# '''
# system_prompt = 
#     Analyze the following underwater image and classify it into one of these categories:
#     0 - algae, sand, rubble, water, blur
#     1 - soft coral or hard coral
#     2 - stable, rocky seafloor without any algae or sand present

#     Provide only the number corresponding to the best class.
# '''

# '''
# system_prompt = 
# You are a specialized agent in classifying underwater images in different classes. Given an 
# image you will classify it in one of the following classes: 0, 1, 2.
# The class 0 corresponds to: algae, sand, rubble, water, blur. 
# The class 1 corresponds to: soft coral or hard coral.
# The class 2 corresponds to: stable, rocky seafloor without any algae or sand present.
# Always provide only the number corresponding to the best representative class and you 
# always have to return one of the values: 0, 1, 2.
# Provide only the number corresponding to the best class.
# '''


# #######################
# #Whole image prompt: 
# ########################
# system_prompt = '''
# You are a specialized agent in classifying underwater images. Your goal is to carefully inspect the image, and then classify it in one of the following classes: 0 or 1.  
# The class 1 corresponds to: rocky, solid seafloor or substrate. 
# The class 0 corresponds to: entirely coral, algae, sand, loose rubble, water or the image is too blurry.

# Pay attention to the image and only classify it as class 0 if you are absolutely certain that it is not class 1.   
# Always provide the output as a dictionary in the format {"class": 0, "conf": 0.5}, where the "class" is an integer number corresponding to the best class: 0 or 1, 
# and the "conf" is a decimal number between 0 and 1 to represent your confidence that the chosen class matches the image. If you think both classes could accurately describe the
# image, the confidence should be closer to 0.
# '''



################
# Patch prompt: 
#################
system_prompt = '''
You are a specialized agent in classifying underwater images. Your goal is to carefully inspect the image, and then classify it in one of the following classes: 0, 1, 2.
The class 1 corresponds to: mainly coral.
The class 2 corresponds to: rocky seafloor or substrate.  It looks solid and has minimal coral.
The class 0 corresponds to: images that do not fit into class 1 or class 2, typically algae, sand, rubble, water or blurry images.
Pay attention to the image and only classify it as class 0 if you're absolutely certain the image cannot be described as class 1 or 2.
Always provide the output as a dictionary in the format {"class": 0, "conf": 0.5}, where the "class" is an integer number corresponding to the best class: 0, 1, 2, 
and the "conf" is a decimal number between 0 and 1 to represent your confidence that the chosen class matches the image. If you think two classes could accurately describe the
image, the confidence should be closer to 0.
'''



class VLMGPT:
	
	def __init__(self):	

		global system_prompt
		self.client = OpenAI(api_key="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")  #"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"
		self.prompt = system_prompt
		print("prompt", self.prompt)

	def __str__(self):
		return f"GPT text parser"
	
	def action(self, image_path: str):  
		try:
			# num_tokens = self._count_tokens()
			# print(num_tokens)

			response = self.client.chat.completions.create(
					   model="gpt-4o", 
                       messages=[   self._text2msg(role="system", text=self.prompt),
						            self._img2msg(role="user", image_path=image_path) ])
			
			# num_tokens = response.usage.prompt_tokens #response.usage.total_tokens #
			# print("input tokens:", num_tokens)
			# out_tokens = response.usage.completion_tokens #response.usage.total_tokens #
			# print("output tokens:", out_tokens)
			
			vlm_answer_str = response.choices[0].message.content
			return self._msg2dict(vlm_answer_str) #self._msg2int(vlm_answer_str) 
			
		except Exception as e: 
			print(f"An error occurred: {str(e)}")
			
		return

	def _encodeImageForOpenAI(self, image_path: str):
		with open(image_path, 'rb') as image_fp:
			base_64_encoded = base64.b64encode(image_fp.read()).decode('utf-8') 
			return base_64_encoded

	def _img2msg(self, role: str, image_path: str):
		return {
            "role": role,
            "content": [
                {"type": "text", "text": "What class is most dominant in the image?"},  # Text prompt
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self._encodeImageForOpenAI(image_path)}","detail": "low"}
                }  # Image
            ]
        }
				
	def _text2msg(self, role: str, text: str):
		return { "role": role, "content": text }

	def _msg2int(self, msg: str):
		try:
			if len(msg) == 1:
				return int(msg)
			else:
				return int(msg[-2])
		except Exception as e: 
			print(f"An error occurred: {str(msg)}")	
			
	def _msg2dict(self, msg: str): 
		msg_dict = json.loads(msg)
		try:
			if isinstance(msg_dict, dict) and "class" in msg_dict and "conf" in msg_dict:
				class_value = int(msg_dict["class"])
				conf_value = float(msg_dict["conf"])
				return class_value, conf_value
			else:
				raise ValueError("Input dictionary does not contain the expected keys: 'class' and 'conf'")
		except Exception as e: 
			print(f"An error occurred: {str(e)}")
			return None, None
	
	def _count_tokens(self, model="gpt-4o"):
		enc = tiktoken.encoding_for_model(model)
		return len(enc.encode(self.prompt))


