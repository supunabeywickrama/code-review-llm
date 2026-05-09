"""
gradio_app.py
-------------
Gradio demo for the Code Review LLM.
Deployable to Hugging Face Spaces for free (set ZeroGPU or CPU space).

Run locally:
    python src/serving/gradio_app.py

Deploy to HF Spaces:
    - Create a new Space (SDK: Gradio)
    - Upload this file as app.py
    - Add MODEL_PATH as a Space secret pointing to your HF Hub model ID
"""

import os
import time
from pathlib import Path

import torch
import gradio as gr
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH     = os.getenv("MODEL_PATH", "./outputs/dpo_model")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are an expert software engineer performing a thorough code review. "
    "Provide specific, actionable feedback on correctness, style, performance, "
    "and security. Be concise but precise."
)

# ── Load model once at startup ─────────────────────────────────────────────────
logger.info(f"Loading {MODEL_PATH} on {DEVICE}...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
_pipe = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    pad_token_id=_tokenizer.eos_token_id,
)
logger.info("Model loaded ✓")

# ── Example snippets ───────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "python",
        """def get_user(user_id):
    conn = sqlite3.connect('users.db')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = conn.execute(query)
    return result.fetchone()""",
    ],
    [
        "javascript",
        """function fetchData(url) {
  return fetch(url)
    .then(response => response.json())
    .then(data => {
      console.log(data)
      return data
    })
}""",
    ],
    [
        "python",
        """class DataProcessor:
    data = []

    def add(self, item):
        self.data.append(item)

    def process(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i] * 2""",
    ],
]


def review_code(language: str, code: str, max_tokens: int) -> tuple[str, str]:
    """Inference function called by Gradio."""
    if not code.strip():
        return "⚠️ Please paste some code to review.", ""

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"### Code to Review:\n```{language}\n{code}\n```\n\n"
        f"### Review:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    t0 = time.perf_counter()
    output = _pipe(prompt, max_new_tokens=max_tokens)[0]["generated_text"]
    latency = round((time.perf_counter() - t0) * 1000)

    if "<|start_header_id|>assistant<|end_header_id|>" in output:
        review = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        review = review.replace("<|eot_id|>", "").strip()
    else:
        review = output[len(prompt):].strip()

    stats = f"⏱ {latency}ms  |  📝 {len(review.split())} words  |  🖥 {DEVICE.upper()}"
    return review, stats


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Code Review LLM",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        """
        # 🔍 Code Review LLM
        **Fine-tuned LLaMA 3.1 8B** with QLoRA + Direct Preference Optimization (DPO)
        for automated, actionable code reviews.

        > Built with: `unsloth` · `trl` · `peft` · `transformers` · `Gradio`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            language = gr.Dropdown(
                choices=["python", "javascript", "typescript", "java", "go", "rust", "c++"],
                value="python",
                label="Language",
            )
            max_tokens = gr.Slider(
                minimum=64, maximum=1024, value=512, step=64,
                label="Max output tokens",
            )
            code_input = gr.Code(
                label="Paste your code here",
                language="python",
                lines=20,
            )
            submit_btn = gr.Button("🔍 Review Code", variant="primary", size="lg")

        with gr.Column(scale=1):
            review_output = gr.Markdown(label="Code Review", value="*Review will appear here...*")
            stats_output  = gr.Textbox(label="Inference stats", interactive=False)

    language.change(fn=lambda lang: gr.Code(language=lang), inputs=language, outputs=code_input)
    submit_btn.click(
        fn=review_code,
        inputs=[language, code_input, max_tokens],
        outputs=[review_output, stats_output],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[language, code_input],
        label="Try these examples",
    )

    gr.Markdown(
        """
        ---
        📦 **Model:** [Hugging Face Hub](https://huggingface.co/your-username/code-review-llm-dpo) &nbsp;|&nbsp;
        💻 **Code:** [GitHub](https://github.com/your-username/code-review-llm) &nbsp;|&nbsp;
        📊 **Training runs:** [W&B](https://wandb.ai/your-username/code-review-llm)
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   # set True to get a public ngrok URL for quick sharing
    )
