# Week 1 Day 6: Generating the Response: Prefill and Decode

On Day 6, we will implement response generation for an LLM chatbot. The implementation is short, but it exercises much
of the code from the previous days. Use this chapter to integrate and debug the complete Week 1 model.

## Task 1: Implement `simple_generate`

```
src/tiny_llm/generate.py
```

`simple_generate` takes a model, tokenizer, prompt, and optional sampler, then streams the generated response to standard
output. Generation has two phases: prefill and decode.

First, implement the nested `_step` function. It takes a one-dimensional array of token IDs, adds the batch dimension,
and passes the result to the model. The model returns unnormalized logits over the vocabulary for every sequence position.

```
y: S (before adding a batch dimension)
model input: 1 x S
output_logits: 1 x S x vocab_size
```

You only need the last token's logits to decide the next token. Therefore, you need to select the last token's logits
from the output logits.

```
logits = output_logits[:, -1, :]
```

You may normalize these logits into log probabilities with the log-sum-exp trick. This normalization does not change
the result of `argmax`, but the sampler introduced on Day 7 expects log probabilities. If `sampler` is `None`, use
`mx.argmax` along the final, vocabulary dimension. Otherwise, pass the log probabilities to `sampler`. Selecting the
highest-scoring token at every step is called greedy decoding.

- 📚 [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- 📚 [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)
- 📚 [Tokenizer definition](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer)

With `_step` complete, implement the rest of `simple_generate`. Begin by encoding the prompt into a one-dimensional token
array with `tokenizer.encode`.

Generate tokens in a loop until the model emits `tokenizer.eos_token_id`. Append each new token to the token array so that
the next model call receives the complete sequence. Feed non-EOS output tokens to `tokenizer.detokenizer`, and print each
new text segment as it becomes available.

An example of the sequences provided to the `_step` function is as below:

```
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode: _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode: _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

In Week 2, we will accelerate decoding with a key-value cache so that the model does not recompute the entire sequence
at every step.

You can test your implementation by running the following command:

```bash
# Start with the default 0.6B model.
hf download Qwen/Qwen3-0.6B-MLX-4bit
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b \
  --prompt "Give me a short introduction to large language model"

# If downloaded, you can also try the larger models.
pdm run main --solution tiny_llm --loader week1 --model qwen3-1.7b \
  --prompt "Give me a short introduction to large language model"
pdm run main --solution tiny_llm --loader week1 --model qwen3-4b \
  --prompt "Give me a short introduction to large language model"
```

Each command should produce a reasonable explanation of large language models. Replace `--solution tiny_llm` with
`--solution ref` to run the reference solution.

{{#include copyright.md}}
