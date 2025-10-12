# Week 1 Day 6: Generating the Response: Prefill and Decode

In day 6, we will implement the process of generating the response when using the LLM as a chatbot. The implementation
is not a lot of code, but given that it uses a large portion of the code we implemented in the previous days, we want
to allocate this day to debug the implementation and make sure everything is working as expected.

# Task 1: Implement `simple_generate`

```
src/tiny_llm/generate.py
```

The `simple_generate` function takes a model, a tokenizer, and a prompt, and generates the response. The generation
process is done in two parts: first prefill, and then decode.

First thing is to implement the `_step` sub-function. It takes a list of tokens `y`. The model will return the logits: the probability distribution of the next token for each position.

```
y: N.. x S, where in week 1 we don't implement batch, so N.. = 1
output_logits: N.. x S x vocab_size
```

You only need the last token's logits to decide the next token. Therefore, you need to select the last token's logits
from the output logits.

```
logits = output_logits[:, -1, :]
```

Then, you can optionally apply the log-sum-exp trick to normalize the logits to avoid numerical instability. As we only
do argmax sampling, the log-sum-exp trick is not necessary. Then, you need to sample the next token from the logits.
You can use the `mx.argmax` function to sample the token with the highest probability over the last dimension
(the vocab_size axis). The function returns the next token number. This decoding strategy is called greedy decoding as we always
pick the token with the highest probability.

- 📚 [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- 📚 [Decoding Strategies in Large Language Models](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)

With the `_step` function implemented, you can now implement the full `simple_generate` function. The function will
first prefill the model with the prompt. As the prompt is a string, you need to first convert it to a list of tokens
by using the tokenizer `tokenizer.encode`.

You will need to implement a while loop to keep generating the response until the model outputs the EOS `tokenizer.eos_token_id` token.
In the loop, you will need to store all previous tokens in a list, and use the detokenizer `tokenizer.detokenizer` to print the response.

An example of the sequences provided to the `_step` function is as below:

```
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode: _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode: _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

We will optimize the `decode` process to use key-value cache to speed up the generation next week.

You can test your implementation by running the following command:

```bash
# Download the models if you haven't done so
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-MLX
huggingface-cli download Qwen/Qwen2-1.5B-Instruct-MLX
huggingface-cli download Qwen/Qwen2-7B-Instruct-MLX
# Run the tests
pdm run main --solution tiny_llm --loader week1 --model qwen2-0.5b \
  --prompt "Give me a short introduction to large language model"
pdm run main --solution tiny_llm --loader week1 --model qwen2-1.5b \
  --prompt "Give me a short introduction to large language model"
pdm run main --solution tiny_llm --loader week1 --model qwen2-7b \
  --prompt "Give me a short introduction to large language model"
```

It should gives you a reasonable response of "what is a large language model". Replace `--solution tiny_llm` with
`--solution ref` to use the reference solution.

{{#include copyright.md}}

