# Trace prompts

The reasoning traces of the training corpus (`reasoning` and `final_answer` in
the corpus, which is not publicly released)
were written by Kimi K3 from these two templates. Each template embeds the same three worked
examples (UniProt P18405, Q53HA8, P50945); a row's `prompt_version` names the template that wrote it.

| | [`v5/`](v5/) | [`v6/`](v6/) |
|---|---|---|
| the teacher writes | the whole trace (five paragraphs), then a Final Answer: summary, InterPro, confident and speculated GO terms, partners | the first paragraph only, grounding the protein in its chain and domain architecture; no GO identifier, no Final Answer |
| corpus rows | `source` `v2` and `v2_extra`: `train`, `test`, `test_old` | `source` `v2_extra_ground`: 37,947 `train` rows, `final_answer` empty |

Use **v5** for a protein with GO labels to reason towards. Use **v6** for a protein whose chain is
well annotated but which has few or no GO labels, where everything after the first paragraph would be
speculation. Both take the same input fields.

## Writing a trace

Rows come from the raw corpus [`wanglab/bioreason-pro-v2`](https://huggingface.co/datasets/wanglab/bioreason-pro-v2):
config `proteins_extra`, split `train` or `test` for v5 and `ground` for v6 (the `v2` rows are config
`proteins`). [`prepare_fields`](../bioreason2/dataset/protein_fields.py) renders every placeholder except
`{examples}`, which is the three example files in name order, joined by a `---` rule.

```python
import glob
import re

import requests
from datasets import load_dataset

from bioreason2.dataset.protein_fields import prepare_fields

VERSION, SPLIT = "v5", "test"                  # "v6", "ground" for a grounding paragraph
row = load_dataset("wanglab/bioreason-pro-v2", "proteins_extra", split=SPLIT)[0]

template = open(f"synth/{VERSION}/{VERSION}_template.md").read()
examples = "\n\n---\n\n".join(open(path).read().strip()
                              for path in sorted(glob.glob(f"synth/{VERSION}/{VERSION}_example*.md")))
prompt = template.format(examples=examples, **prepare_fields(row))

response = requests.post(
    "https://<your-endpoint>/v1/chat/completions",
    headers={"Authorization": "Bearer <your-key>"},
    json={"model": "<Kimi K3 at your endpoint>",
          "messages": [{"role": "user", "content": prompt}],      # one user turn, no system prompt
          "max_tokens": 32000, "reasoning_effort": "high"},       # no temperature or top_p
    timeout=900,
).json()
text = response["choices"][0]["message"]["content"]
reasoning = re.search(r"<\|REASONING_START\|>(.*?)(?:<\|REASONING_END\|>|<\|FINAL_ANSWER_START\|>|\Z)", text, re.S)
answer = re.search(r"<\|FINAL_ANSWER_START\|>(.*?)(?:<\|FINAL_ANSWER_END\|>|\Z)", text, re.S)   # v5 only
```

- **The teacher sees the labels; the student does not.** The prompt carries the protein's GO labels,
  name, function summary, UniProt comments and full residue-feature annotation, and the template has
  the teacher derive all of it from the chain and organism, so the trace is one the student can learn.
- **The fill is exact.** It reproduces byte for byte the prompts sent for the `v2_extra` and
  `v2_extra_ground` rows. About 6% of `v2` rows differ in the feature block: those traces predate a
  renderer fix that stopped repeating a position within a scattered-site bullet.
- **Effort.** Check that your server honours the effort field; some accept only the nested
  `"reasoning": {"effort": "high"}` and ignore the flat one without an error. Allow up to ten minutes
  per call.
- **The corpus traces passed checks before release.** A trace that lost a tag or an answer section, put
  a GO identifier in the reasoning, cited an InterPro accession absent from its input, or referred to
  the input itself ("as annotated") was regenerated, and dropped if it failed again. The Final Answer
  was then repaired against the row (GO identifiers and names against `go-basic.obo`, confident terms
  set to the supplied ones, omitted InterPro entries and partners restored); the reasoning is as written.
