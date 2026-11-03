# Model attributions

Use does not imply endorsement by upstream creators.

## ModernBERT

- **Base model:** `answerdotai/ModernBERT-base`.
- **Creators:** Answer.AI, LightOn and the ModernBERT authors/contributors.
- **Model source:** <https://huggingface.co/answerdotai/ModernBERT-base>.
- **Code source:** <https://github.com/AnswerDotAI/ModernBERT>.
- **License:** Apache License, Version 2.0 (SPDX: `Apache-2.0`).
- **Upstream license and notices:**
  [ModernBERT-Apache-2.0.txt](ModernBERT-Apache-2.0.txt), copied from
  <https://raw.githubusercontent.com/AnswerDotAI/ModernBERT/main/LICENSE>.
- **Citation:** Warner et al. (2024), *Smarter, Better, Faster, Longer: A Modern
  Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning
  and Inference*, <https://arxiv.org/abs/2412.13663>.

**Use and modifications:** acclass fine-tunes canonical ModernBERT-base revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8` as a nine-head shell-command and
action-description classifier. The classifier, folded calibration temperature,
thresholds and Q8/Q5 exports are acclass modifications, not unmodified upstream checkpoints. Measured results and limitations are in
[`docs/model-card.md`](../docs/model-card.md).

The upstream model remains subject to Apache 2.0. The copied code-repository
license preserves its existing copyright notice verbatim.
