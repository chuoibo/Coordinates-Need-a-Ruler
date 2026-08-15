# LlamaFactory files carrying our changes

Training runs on [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) by
Yaowei Zheng et al. (Apache-2.0). We are grateful for it: the multimodal data
pipeline, the Qwen templates and the LoRA plumbing here are theirs, and this
project would have been a much larger undertaking without them.

```bibtex
@inproceedings{zheng2024llamafactory,
  title     = {LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author    = {Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and
               Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for
               Computational Linguistics (Volume 3: System Demonstrations)},
  year      = {2024},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics}
}
```

`files/` holds the seven files we modified, verbatim, at **LlamaFactory
0.9.5.dev0**. They keep LlamaFactory's Apache-2.0 headers. Install them with:

```bash
python scripts/apply_llamafactory_patch.py --llamafactory /path/to/LLaMA-Factory
```

## What each file contributes

| File | Change |
| --- | --- |
| `train/trainer_utils.py` | `ntl_loss_func` -- Eq. (4)-(6). Restricted softmax over the coordinate columns, Wasserstein-1 to the target bin, summed under the same denominator as cross-entropy. |
| `train/sft/trainer.py` | Resolves the `<coord_0>..<coord_1000>` block on the tokenizer, refuses to start if it is not contiguous and ordered, binds the loss, and registers the gradient-mask callback. |
| `train/callbacks.py` | `CoordEmbeddingGradMaskCallback` -- Eq. (3). Hooks every trainable 2-D embedding/head parameter and zeroes gradient outside the block. |
| `model/model_utils/embedding.py` | Description-based initialisation -- Eq. (2) -- plus the Gaussian comparison arm, both addressing rows by id rather than by position. |
| `hparams/finetuning_args.py` | `use_ntl_loss`, `ntl_loss_weight`, `ntl_coord_token_prefix`, `train_only_new_token_embeddings`. |
| `hparams/model_args.py` | `init_special_tokens` accepts `desc_init` / `desc_init_w_noise` / `gaussian_init`. |
| `data/template.py` | The `qwen3_5_nothink` template: the model emits the JSON object directly, with no `<think>` block. |

## Reading the algorithms without LlamaFactory

`src/cnr/` carries standalone, unit-tested implementations of the same three
pieces -- `cnr.ntl`, `cnr.desc_init`, `cnr.grad_mask` -- with no framework
dependency. They are the readable reference; these files are the ones that
actually ran.
