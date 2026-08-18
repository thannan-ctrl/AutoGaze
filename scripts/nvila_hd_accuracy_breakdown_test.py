"""Fine-grained latency breakdown: dense vs AutoGaze (chunked-batched) on the
same EgoSchema sample, split by CPU/GPU stage and LLM prefill/decode.

See README.md for what each measured stage means, the exact command, and
results. Implementation lives in scripts/breakdown/ (config, timing,
instrumentation, dataset, processor, runner, summary, main).
"""
from breakdown.main import main

if __name__ == "__main__":
    main()
