import numpy as np
import gradio as gr
from tts_reference import TTS

def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)

def greet(name):
    return "Hello " + name + "!"


html_str = """

    <h1 style='font-family: "Nunito",sans-serif; color: midnightblue !important; font-size: 2.1875rem;margin:20px;text-align: center !important;'>AI Content Generate Demo</h1>
    <h3 style='color: midnightblue !important; font-size: 1.1875rem;text-align: center !important;'> this is a demo that show the artificial intelligent generate content</h3>
"""

def text_to_wav(model_path,
                json_path,
                text):

    text = text.replace("\n", ",")
    text = text.replace(" ", "")

    print(model_path)
    print(json_path)
    print(text)

    sr, audio  = TTS(model_path=model_path,
               json_path = json_path,
               text=text)

    return sr ,audio


with gr.Blocks() as demo:

    ht = gr.HTML(html_str, visible=True)
    
    with gr.Tab("VITS"):

        gr.Interface(fn=text_to_wav,
                     inputs=["text","text", "text"],
                     outputs=["audio"],
                     description="VITS推理界面")

    with gr.Tab("Flip Text"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")

    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")



    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")


    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()

