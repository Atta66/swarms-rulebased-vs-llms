import matplotlib.pyplot as plt

# Data: tokens and corresponding times (seconds)
tokens = [582, 330, 75]
times_qwen = [1.836, 1.157, 0.847]
times_gpt = [1.397, 0.761, 0.630]

plt.figure(figsize=(6,4))
plt.plot(tokens, times_qwen, marker='o', linestyle='-', color='b', label='Qwen2.5 (Local LLM)')
plt.plot(tokens, times_gpt, marker='s', linestyle='--', color='r', label='ChatGPT')
plt.xlabel('Tokens')
plt.ylabel('Response Time (seconds)')
plt.title('Tokens vs Response Time Latency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
