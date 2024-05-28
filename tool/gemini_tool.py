import time
import pathlib
import textwrap

import google.generativeai as genai

from PIL import Image


def gemini_generate(img_path, prompt):
    img = Image.open(img_path)
    genai.configure(api_key='AIzaSyABDU1Vhuj9F_VlLQ4to0t7q3DDaYhcuJo')
    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]
    model = genai.GenerativeModel('gemini-pro-vision',safety_settings=safety_settings)
    response = model.generate_content(['Based on this image of a meme, answer the following question:'+prompt,img])
    return response.text

