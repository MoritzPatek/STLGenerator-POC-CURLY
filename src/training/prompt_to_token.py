
def tokenize_prompt(prompt, tokenizer):
    """
    This function tokenizes the prompt using the provided tokenizer.

    Args:
        prompt: str
        tokenizer: GPT2Tokenizer

    Returns:
        token_ids: List[int]
    """

    # Tokenize the prompt
    tokens = tokenizer.tokenize(prompt)

    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids
