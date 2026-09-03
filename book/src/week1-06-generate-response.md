# Week 1 Day 6: Generating the Response: Prefill and Decode

On Day 6, we will implement response generation for an LLM chatbot. The implementation is short, but it exercises much
of the code from the previous days. Use this chapter to integrate and debug the complete Week 1 model.

## Task 1: Implement `simple_generate`

```
src/tiny_llm/generate.py
```

`simple_generate` takes a model, tokenizer, prompt, and optional sampler, then streams the generated response to standard
output. Its optional `max_tokens` argument defaults to 256 and limits the number of newly generated tokens. Generation
has two phases: prefill and decode.

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
`mx.argmax` along the final, vocabulary dimension. Otherwise, pass the `(1, vocab_size)` log-probability array to
`sampler`, which returns one token ID in an integer array with shape `(1,)`. Selecting the highest-scoring token at every
step is called greedy decoding.

- 📚 [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- 📚 [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)
- 📚 [Tokenizer definition](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer)

With `_step` complete, implement the rest of `simple_generate`. Begin by encoding the prompt into a one-dimensional token
array with `tokenizer.encode(prompt, add_special_tokens=False)`. `main.py` has already formatted the chat prompt, so the
tokenizer must not add another set of special tokens. If encoding produces no tokens, reject the prompt before calling
the model.

Generate tokens in a loop until the model emits `tokenizer.eos_token_id` or `max_tokens` new tokens have been produced.
Append each new token to the token array so that the next model call receives the complete sequence. An EOS token already
inside the prompt is context and does not stop generation; only a newly generated EOS does.

Before the loop, bind one streaming detokenizer with `detokenizer = tokenizer.detokenizer`, then call
`detokenizer.reset()` once. Keep and reuse that same stateful object throughout generation: feed every non-EOS output
token to `detokenizer.add_token(...)` and print each `detokenizer.last_segment` as it becomes available. With the locked
mlx-lm 0.31.3 dependency, each separate access to the `tokenizer.detokenizer` property creates a fresh streaming
detokenizer, so repeatedly accessing the property would discard the buffered text. On either termination path, call
`detokenizer.finalize()` on the saved object and print its final `last_segment` so buffered text is not lost. The
function returns `None` after streaming the response.

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

First run the deterministic checkpoint, which does not download or load a model:

```bash
pdm run test --week 1 --day 6
```

Then complete the required product check with the default 0.6B model:

```bash
hf download Qwen/Qwen3-0.6B-MLX-4bit
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b \
  --prompt "Give me a short introduction to large language model"
```

The first command downloads the model once; later runs use the cached copy. The product command should produce a
reasonable explanation of large language models. Replace `--solution tiny_llm` with `--solution ref` to run the
reference solution.

If downloaded, you can also try the larger models; these are optional demonstrations, not completion requirements:

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-1.7b \
  --prompt "Give me a short introduction to large language model"
pdm run main --solution tiny_llm --loader week1 --model qwen3-4b \
  --prompt "Give me a short introduction to large language model"
```

{{#include copyright.md}}
