import gradio as gr

def test_click():
    print("Minimal test_click function CALLED!")
    return "Minimal button was clicked! (check console)"

with gr.Blocks() as demo:
    gr.Markdown("Super Simple Gradio Button Test")
    btn = gr.Button("Test Me")
    status = gr.Textbox(label="Status")
    btn.click(test_click, [], [status])

print("Launching minimal Gradio test app...")
demo.launch()
print("Minimal Gradio test app finished or closed.") 