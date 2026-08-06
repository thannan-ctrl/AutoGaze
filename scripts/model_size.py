"""Print parameter counts for every component of AutoGaze."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

from autogaze.models.autogaze import AutoGaze


def param_count(module):
    total = sum(p.numel() for p in module.parameters())
    unique = sum(p.numel() for p in set(module.parameters()))
    return total, unique


def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.3f}B"
    if n >= 1e6:
        return f"{n/1e6:.3f}M"
    return f"{n/1e3:.1f}K"


def print_table(rows, title):
    col_w = [max(len(r[i]) for r in rows + [(title, "", "")]) for i in range(3)]
    sep = "─" * (col_w[0] + col_w[1] + col_w[2] + 10)
    print(f"\n{title}")
    print(sep)
    print(f"  {'Component':<{col_w[0]}}  {'Params':>{col_w[1]}}  {'Unique':>{col_w[2]}}")
    print(sep)
    for name, params, unique in rows:
        marker = "  " if params == unique else "* "
        print(f"{marker}{name:<{col_w[0]}}  {params:>{col_w[1]}}  {unique:>{col_w[2]}}")
    print(sep)
    print("  * shared parameters (unique count differs from total)")


print("Loading nvidia/AutoGaze ...")
model = AutoGaze.from_pretrained("nvidia/AutoGaze")
gm = model.gazing_model

# ── Top-level AutoGaze ────────────────────────────────────────────────────────
total_p, total_u = param_count(model)

rows_top = [
    ("gazing_model (total)", fmt(param_count(gm)[0]),        fmt(param_count(gm)[1])),
    ("  vision_model",       fmt(param_count(gm.vision_model)[0]),  fmt(param_count(gm.vision_model)[1])),
    ("    temporal_conv",    fmt(param_count(gm.vision_model.temporal_conv)[0]), fmt(param_count(gm.vision_model.temporal_conv)[1])),
    ("    norm",             fmt(param_count(gm.vision_model.norm)[0]),          fmt(param_count(gm.vision_model.norm)[1])),
    ("    blocks",           fmt(param_count(gm.vision_model.blocks)[0]),        fmt(param_count(gm.vision_model.blocks)[1])),
    ("    out_proj",         fmt(param_count(gm.vision_model.out_proj)[0]),      fmt(param_count(gm.vision_model.out_proj)[1])),
    ("  connector",          fmt(param_count(gm.connector)[0]),     fmt(param_count(gm.connector)[1])),
    ("  gaze_decoder",       fmt(param_count(gm.gaze_decoder)[0]),  fmt(param_count(gm.gaze_decoder)[1])),
    ("    embed_tokens",     fmt(param_count(gm.gaze_decoder.model.embed_tokens)[0]), fmt(param_count(gm.gaze_decoder.model.embed_tokens)[1])),
    ("    layers",           fmt(param_count(gm.gaze_decoder.model.layers)[0]),  fmt(param_count(gm.gaze_decoder.model.layers)[1])),
    ("    lm_head",          fmt(param_count(gm.gaze_decoder.lm_head)[0]),       fmt(param_count(gm.gaze_decoder.lm_head)[1])),
]

print_table(rows_top, "AutoGaze component breakdown")

# ── Per-layer gaze decoder ────────────────────────────────────────────────────
rows_layers = []
for i, layer in enumerate(gm.gaze_decoder.model.layers):
    p, u = param_count(layer)
    rows_layers.append((f"layer {i:02d}", fmt(p), fmt(u)))
rows_layers.append(("TOTAL layers", fmt(param_count(gm.gaze_decoder.model.layers)[0]), fmt(param_count(gm.gaze_decoder.model.layers)[1])))

print_table(rows_layers, "Gaze decoder — per transformer layer")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nTotal AutoGaze parameters : {fmt(total_p)}  (unique: {fmt(total_u)})")
cfg = model.config.gaze_model_config
print(f"Patch vocab size          : {cfg.gaze_decoder_config.vocab_size}  ({cfg.num_vision_tokens_each_frame} patches + 1 EOS)")
print(f"Scales                    : {model.scales}")
print(f"Patches per scale         : {model.num_vision_tokens_each_scale_each_frame}")
print(f"Input image size          : {cfg.input_img_size}px")
