# Offline tokenizer data

The direct inference worker implements tokenization itself. It needs four fixed
vocabulary tables, fetched once by `npm run setup:image` and verified against
pinned SHA-256 hashes:

- OpenAI CLIP BPE vocabulary (MIT)
- Qwen3-0.6B-Base vocabulary and merges (Apache-2.0)
- T5 tokenizer table (Apache-2.0)

These are model data, not executable libraries. Image generation never contacts
the network, and no prompts, images, histories, or generated metadata are stored
in this directory.
