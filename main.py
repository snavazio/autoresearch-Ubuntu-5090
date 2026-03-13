import os, subprocess, time
from openai import OpenAI

# --- CONFIG ---
MODEL_NAME = "llama3-8k" 
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def run_train():
    print("\n🧹 Cleaning up GPU memory and old processes...")
    # Kill any hung python processes from previous failed runs
    subprocess.run(["pkill", "-9", "python3"], stderr=subprocess.DEVNULL)
    
    # Give the GPU a second to release memory
    time.sleep(2)
    
    print("🚀 Starting experiment on Thing 2...")
    # Run the training script and capture all output
    result = subprocess.run(
        ["uv", "run", "train.py"], 
        text=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT
    )
    
    output = result.stdout
    
    # Check for hardware-specific crashes (like OOM or NameErrors)
    if result.returncode != 0 or "Error" in output or "OutOfMemoryError" in output:
        print("❌ CRASHED or OOM.")
        # Return the last chunk of logs so the AI can see why it failed
        return f"CRASH_ERROR: {output[-1000:]}", output
    
    # Look for the loss in the output to report back to the LLM
    for line in reversed(output.split('\n')):
        if "Loss" in line: 
            return line, output
            
    return "SUCCESS: No metric found.", output
def get_next_iteration(last_log):
    with open("program.md", "r") as f: instructions = f.read()
    with open("train.py", "r") as f: current_code = f.read()

    prompt = f"""You are an AI ML Lead. FIX CRASHES FIRST.
    
    CRITICAL HARDWARE CONSTRAINTS FOR RTX 5090 (Thing 2):
    1. DO NOT change the 'apply_rotary_emb' function or the 'cos/sin' 4D buffer registration.
    2. KEEP the '.transpose(1, 2)' calls in CausalSelfAttention; these are required for Blackwell SDPA.
    3. KEEP DEVICE_BATCH_SIZE at 32. Do not exceed this or it will trigger a CUDA Out of Memory error.
    4. DO NOT define a custom Tokenizer class. Use the one from prepare.py.
    5. KEEP the "window_pattern": "L" and the current GPT architecture (8 layers, 512 embd).
    
    GOAL: {instructions}
    LOGS: {last_log}
    CODE: {current_code}
    
    Respond ONLY with the full updated train.py in a single code block. 
    Maintain all existing hardware optimizations for VRAM and speed.
    """
    
    print(f"🧠 Asking {MODEL_NAME} for new hyperparameters...")
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, extra_body={"num_ctx": 8192}
    )
    return response.choices[0].message.content

def main():
    while True:
        metric, full_log = run_train()
        print(f"📊 Result: {metric}")
        
        raw_suggestion = get_next_iteration(full_log)
        new_code = raw_suggestion.split("```python")[-1].split("```")[0].strip()

        # --- THE SAFETY LOCK ---
        # Force re-injection of the hardware setup and imports if the LLM deleted them
        if "from prepare import" not in new_code:
            print("🛡️ Safety: Re-injecting missing imports...")
            new_code = "from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb\n" + new_code
            
        if "device =" not in new_code:
            new_code = "import torch\ndevice = 'cuda'\ntorch.backends.cuda.enable_flash_sdp(False)\n" + new_code

        # Force kill the 'Toy Model' hallucination if it appears
        if "class Tokenizer:" in new_code:
            print("🛡️ Safety: Removing hallucinated Tokenizer class...")
            # Simple way to strip the toy class the LLM keeps writing
            import re
            new_code = re.sub(r"class Tokenizer:.*?def get_vocab_size\(self\):.*?return self\.vocab_size", "", new_code, flags=re.DOTALL)

        with open("train.py", "w") as f: f.write(new_code)
        print("📝 train.py updated.")
        time.sleep(2)

if __name__ == "__main__":
    main()