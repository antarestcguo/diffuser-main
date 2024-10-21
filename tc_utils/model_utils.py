import torch
from PIL import Image
import os

def text2embedding(tokenizer, text_encoder, prompt_tokens, max_length=20, b_remove_pad=True):
    # token
    text_inputs = tokenizer(
        prompt_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt_tokens, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1: -1]
        )
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
    attention_mask = None
    prompt_embeds = text_encoder(
        text_input_ids.to("cuda"),
        attention_mask=attention_mask,
    )
    if b_remove_pad:
        prompt_embeds = prompt_embeds[0][text_input_ids != 0].unsqueeze(0)
    else:
        prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def img2embedding(clip_model, processor, img):
    import pdb
    pdb.set_trace()

    img_inputs = processor(images=img, return_tensors="pt").to("cuda")
    image_features = clip_model.get_image_features(**img_inputs)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features


def resizeImg(img):
    basesize = 512

    hsize = int(img.size[1])
    wsize = int(img.size[0])

    max_size = max(hsize, wsize)

    if max_size == hsize:
        resize_h = basesize
        resize_w = int(float(basesize) / hsize * wsize)
    else:
        resize_w = basesize
        resize_h = int(float(basesize) / wsize * hsize)

    resize_img = img.resize((resize_w, resize_h), Image.ANTIALIAS)
    return resize_img


def load_textual_embedding(file_path, tokenizer, text_encoder, default_file_name="learned_embeds.bin"):
    # load file
    state_dict = torch.load(os.path.join(file_path, default_file_name), map_location="cpu")
    loaded_token, embedding = next(iter(state_dict.items()))

    token = loaded_token
    embedding = embedding.to(dtype=text_encoder.dtype, device=text_encoder.device)

    vocab = tokenizer.get_vocab()
    if token in vocab:
        raise ValueError(
            f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
        )
    elif f"{token}_1" in vocab:
        multi_vector_tokens = [token]
        i = 1
        while f"{token}_{i}" in tokenizer.added_tokens_encoder:
            multi_vector_tokens.append(f"{token}_{i}")
            i += 1

        raise ValueError(
            f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
        )

    is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

    if is_multi_vector:
        tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
        embeddings = [e for e in embedding]  # noqa: C416
    else:
        tokens = [token]
        embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

    # add tokens and get ids
    tokenizer.add_tokens(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # resize token embeddings and set new embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    for token_id, embedding in zip(token_ids, embeddings):
        text_encoder.get_input_embeddings().weight.data[token_id] = embedding

    print(f"Loaded textual inversion embedding for {token}.")

    return tokenizer, text_encoder
