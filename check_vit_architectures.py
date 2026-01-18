#!/usr/bin/env python3
"""
Check if the ViT models have exactly the same architecture.
"""
import torch
from transformers import ViTForImageClassification, ViTConfig
import gc

models_to_check = [
    "google/vit-base-patch16-224",
    "google/vit-base-patch16-224-in21k",
    "timm/vit_base_patch16_224.dino",
    "timm/vit_base_patch16_224.mae",
    "timm/vit_base_patch16_clip_224.openai",
    "timm/vit_base_patch16_clip_224.laion2b",
]

print("=" * 70)
print(" CHECKING VIT ARCHITECTURES")
print("=" * 70)

all_param_names = {}
all_param_shapes = {}
all_configs = {}

for model_name in models_to_check:
    print(f"\n{model_name}:")
    try:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )

        # Get config
        config = model.config
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_hidden_layers: {config.num_hidden_layers}")
        print(f"  num_attention_heads: {config.num_attention_heads}")
        print(f"  intermediate_size: {config.intermediate_size}")
        print(f"  patch_size: {config.patch_size}")
        print(f"  image_size: {config.image_size}")

        all_configs[model_name] = {
            'hidden_size': config.hidden_size,
            'num_hidden_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'intermediate_size': config.intermediate_size,
            'patch_size': config.patch_size,
            'image_size': config.image_size,
        }

        # Get parameter names and shapes
        param_names = []
        param_shapes = []
        total_params = 0
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'pooler' not in name:
                param_names.append(name)
                param_shapes.append(tuple(param.shape))
                total_params += param.numel()

        all_param_names[model_name] = param_names
        all_param_shapes[model_name] = param_shapes

        print(f"  Total encoder params: {total_params:,}")
        print(f"  Num param tensors: {len(param_names)}")

        del model
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")

# Compare architectures
print("\n" + "=" * 70)
print(" ARCHITECTURE COMPARISON")
print("=" * 70)

reference = models_to_check[0]
ref_names = all_param_names.get(reference, [])
ref_shapes = all_param_shapes.get(reference, [])
ref_config = all_configs.get(reference, {})

for model_name in models_to_check[1:]:
    print(f"\n{reference} vs {model_name}:")

    names = all_param_names.get(model_name, [])
    shapes = all_param_shapes.get(model_name, [])
    config = all_configs.get(model_name, {})

    # Check config
    config_match = ref_config == config
    print(f"  Config match: {config_match}")
    if not config_match:
        for key in ref_config:
            if ref_config.get(key) != config.get(key):
                print(f"    {key}: {ref_config.get(key)} vs {config.get(key)}")

    # Check param names
    names_match = ref_names == names
    print(f"  Param names match: {names_match}")
    if not names_match:
        missing = set(ref_names) - set(names)
        extra = set(names) - set(ref_names)
        if missing:
            print(f"    Missing: {list(missing)[:5]}...")
        if extra:
            print(f"    Extra: {list(extra)[:5]}...")

    # Check param shapes
    shapes_match = ref_shapes == shapes
    print(f"  Param shapes match: {shapes_match}")
    if not shapes_match and len(ref_shapes) == len(shapes):
        for i, (s1, s2) in enumerate(zip(ref_shapes, shapes)):
            if s1 != s2:
                print(f"    {ref_names[i]}: {s1} vs {s2}")

print("\n" + "=" * 70)
print(" SUMMARY")
print("=" * 70)

# Check if all configs are the same
configs_list = list(all_configs.values())
all_same_config = all(c == configs_list[0] for c in configs_list)
print(f"\nAll models have same config: {all_same_config}")

# Check if all param structures are the same
all_same_structure = all(
    all_param_names.get(m) == ref_names and all_param_shapes.get(m) == ref_shapes
    for m in models_to_check
)
print(f"All models have same param structure: {all_same_structure}")
