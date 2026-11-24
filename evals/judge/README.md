# LLM judge

Fmax answers one question: were the GO identifiers right? It cannot see that a model reached the right
identifier by copying its context, or that it wrote paragraphs of confident mechanism around a domain
that does not exist. The judge scores the written answer on five axes against the protein's own
annotation. It complements Fmax rather than replacing it.

- [`judge_prompt.py`](judge_prompt.py): the prompt, the verdict schema and the axes. This is the part to
  reuse with any judge model.
- [`llm_judge.py`](llm_judge.py): a runner for any OpenAI-compatible `/chat/completions` endpoint. To use
  another API, replace `call_judge`.

## The protocol

Every judge number in the README was produced this way; a run that deviates is not comparable with them.

```bash
# 1. join the generations to the corpus on the pinned 1,000-protein sample
python evals/judge/llm_judge.py --export-rows rows.jsonl --evals-dir results/rl \
    --ids-file data/eval/judge_subset.txt --test-dataset data/v2/proteins

# 2. judge: Nemotron 3.5 Super VL at reasoning effort max
export JUDGE_BASE_URL=<https://your-endpoint/v1> JUDGE_API_KEY=<key>
python evals/judge/llm_judge.py --rows-jsonl rows.jsonl --out judge/rl \
    --model <Nemotron 3.5 Super VL at your endpoint> --effort max --max-output-tokens 32000 --workers 96

# 3. report
python evals/judge/llm_judge.py --summarize --out judge/rl
```

`--test-dataset` is the corpus as `scripts/data/fetch_corpus.py` writes it, not a prompt table: the judge
needs the label columns. The InterPro section the judge reads comes from the corpus's
`interpro_formatted`, which `wanglab/bioreason-pro-v2` renders differently from the copy the published
numbers were judged with (the same entries, merged spans and their types). A run
writes one JSON per protein (resumable; existing files are skipped unless `--overwrite`) with the parsed
verdict, the raw output, the judge's reasoning and token usage, and makes up to three passes
(`--passes`), re-judging any verdict that misses an axis; one still incomplete after the last pass is
reported, and `<out>/_retry/` keeps the replaced verdicts. `--summarize` leaves out verdicts written
without reasoning (`usage.reasoning_tokens == 0`). Compare models on the proteins judged for all of
them. `--dry-run` builds every prompt without calling; `--print-prompt` prints the first.

## The rubric

Five axes, each 1-10 (or -1 when not applicable): molecular function, biological process, cellular
component, specificity, and reliability. The judge first states what the protein does from the
evidence, then scores the answer's prose against it, reading GO terms as biology rather than
identifiers to tick off.

The judge sees what the trace teacher saw (see [`synth/`](../../synth/)): the protein's name,
accession, organism, length, UniProt summary and comments, residue-feature annotation, numbered
sequence, interaction partners and GO terms. The feature block and the numbered sequence let it check
positional and residue-identity claims; without them it marks correct coordinates as invented. GO-GPT's
predictions are never shown, since rewarding agreement with the model's own input would reward copying.

## Traps

- **Throughput is concurrency.** A call takes 30-55 s, so 8 workers give ~4 s per protein and 96
  give well under one.
- **Scores compare only within one judge, effort and rubric,** and best within one judging window:
  the same answers scored up to a third of a point apart weeks apart.
- **Exit code 2** means at least one protein errored; re-run to fill the gap.
- **Generations' `<think>` tags are renamed `<reasoning>`**, because Nemotron reads its own reasoning
  tags even inside user text.
