"""Append three Student XYZ features without changing initial Actor outputs."""

import argparse
import copy
from pathlib import Path

import torch


def expand_checkpoint(source: dict) -> dict:
    result = copy.deepcopy(source)
    actor = result["actor_state_dict"]
    weight = actor["mlp.0.weight"]
    if weight.shape != (512, 261) or actor["obs_normalizer._mean"].shape != (1, 133):
        raise ValueError("Expected the 133-current + 128-latent Intent checkpoint.")

    def expand_weight(value):
        return torch.cat((value[:, :133], value.new_zeros(512, 3), value[:, 133:]), dim=1)

    actor["mlp.0.weight"] = expand_weight(weight)
    for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        key = f"obs_normalizer.{name}"
        value = actor[key]
        if value.shape != (1, 133):
            raise ValueError(f"Unexpected normalizer shape: {key}")
        actor[key] = torch.cat((value, value.new_full((1, 3), fill)), dim=-1)

    # Only the Actor first-layer Adam state has this shape in this architecture.
    matched = 0
    for state in result["optimizer_state_dict"]["state"].values():
        moment = state.get("exp_avg")
        if moment is not None and moment.shape == weight.shape:
            matched += 1
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    if state[key].shape != weight.shape:
                        raise ValueError("Inconsistent Adam moment shape.")
                    state[key] = expand_weight(state[key])
    if matched != 1:
        raise ValueError(f"Expected exactly one first-layer Adam state, found {matched}.")
    infos = dict(result.get("infos") or {})
    infos["global_root_input_migration"] = {
        "old_current_dim": 133, "new_current_dim": 136,
        "new_input_weights": "zero", "optimizer": "preserved; new moments zero",
    }
    result["infos"] = infos
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = expand_checkpoint(checkpoint)
    # Verify the input insertion does not disturb the latent columns.
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(32, 261, generator=generator)
    expanded = torch.cat((x[:, :133], torch.randn(32, 3, generator=generator), x[:, 133:]), dim=1)
    torch.testing.assert_close(
        x @ checkpoint["actor_state_dict"]["mlp.0.weight"].T,
        expanded @ migrated["actor_state_dict"]["mlp.0.weight"].T,
        atol=2e-5, rtol=2e-5,
    )
    torch.save(migrated, args.output)
    print(f"Saved {args.output}: current=136, ONNX=631, action=29; initial Actor mapping verified")


if __name__ == "__main__":
    main()
