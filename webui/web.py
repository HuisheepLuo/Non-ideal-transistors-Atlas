import gradio as gr
import os
import sys
sys.path.append(os.getcwd())

from webui.gradio_set.params import iface_lib, param_input
from webui.gradio_set.fitting import iface_fit, param_fit_input


iface1 = iface_lib()
iface2 = iface_fit()

with gr.Blocks() as demo:
    with gr.Tab(label="Non-ideal Model Atlas"):
        # Tab 1
        gr.Markdown(iface1.describe())
        with gr.Column():
            with gr.Row():
                inps = iface1.ui()
            with gr.Row():
                outps = gr.Gallery().style(
                    grid=[4], 
                    height="auto"
                )
        title = iface1.title()
        btn = gr.Button("Run")
        btn.click(fn=param_input, inputs=inps, outputs=outps)
    
    with gr.Tab(label="Fitting"):
        # Tab 2
        gr.Markdown(iface2.describe())
        with gr.Column():
            with gr.Column():
                inps = iface2.ui()
            with gr.Column():
                outps = [
                    gr.Textbox(label='Extract value'),
                    gr.Gallery().style(
                        grid=[5], 
                        height="500", 
                        container=True
                    )
                ]
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
            [
                1000, 10, 1e-8, 40, 2,
                2, 10, 0.1, 'Transfer', False, False, False, True, 'DTW', 'p-type', 
                [
                    [0, 0.02, 0.001, 0.005],
                    [0.1, 0.3, 0.06, 0.02],
                    [0.3, 0.6, 0.1, 0.5],
                ], 
                os.path.join(os.path.dirname(__file__), "transfer_example.csv")
            ],
            [
                1000, 10, 1e-8, 40, 20,
                20, 10, 0.1, 'Transfer', True, False, False, True, 'DTW', 'n-type', 
                [
                    [0, 0.02, 0.001, 0.005],
                    [0.1, 0.3, 0.06, 0.02],
                    [0.3, 0.6, 0.1, 0.5],
                ], 
                os.path.join(os.path.dirname(__file__), "transfer_example_k.csv")
            ],
            ],
            inputs=inps,
            outputs=outps,
            fn=param_fit_input,
            cache_examples=False,
        )
        title = iface2.title()
        btn = gr.Button("Fit")
        btn.click(fn=param_fit_input, inputs=inps, outputs=outps)

if __name__ == '__main__':
    demo.launch(enable_queue=True)
    