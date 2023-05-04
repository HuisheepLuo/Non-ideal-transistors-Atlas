import gradio as gr
import os
import sys
sys.path.append(os.getcwd())

from webui.gradio_set.params import interface_lib, param_input
from webui.gradio_set.match import interface_fit, param_fit_input


intf_lib = interface_lib()
intf_fit = interface_fit()

with gr.Blocks() as demo:
    with gr.Tab(label="Atlas"):
        # Tab 1
        gr.Markdown(intf_lib.describe())
        with gr.Column():
            with gr.Column():
                inps = intf_lib.ui()
            with gr.Row():
                outps = gr.Gallery().style(
                    grid=[4], 
                    height="auto"
                )
        title = intf_lib.title()
        btn = gr.Button("Run")
        btn.click(fn=param_input, inputs=inps, outputs=outps)
    
    with gr.Tab(label="Matching"):
        # Tab 2
        gr.Markdown(intf_fit.describe())
        with gr.Column():
            with gr.Column():
                inps = intf_fit.ui()
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
                os.path.join(os.path.dirname(__file__), "example_curves\\transfer_example.csv")
            ],
            [
                1000, 10, 1e-8, 40, 20,
                20, 10, 0.1, 'Transfer', True, False, False, True, 'DTW', 'n-type', 
                [
                    [0, 0.02, 0.001, 0.005],
                    [0.1, 0.3, 0.06, 0.02],
                    [0.3, 0.6, 0.1, 0.5],
                ], 
                os.path.join(os.path.dirname(__file__), "example_curves\\transfer_example_k.csv")
            ],
            ],
            inputs=inps,
            outputs=outps,
            fn=param_fit_input,
            cache_examples=False,
        )
        title = intf_fit.title()
        btn = gr.Button("Fit")
        btn.click(fn=param_fit_input, inputs=inps, outputs=outps)

if __name__ == '__main__':
    demo.launch(enable_queue=True)
    