import os

from awesome_nlp.utils import text_attention_heatmap


class Visualizer:
    def __init__(
        self,
        sample_ids=None,
        single_file=False,
        output_dir=None,
        exit_after_vis=False,
    ):
        self.single_file = single_file
        self.output_dir = output_dir
        self.exit_after_vis = exit_after_vis

        if isinstance(sample_ids, int):
            self.sample_ids = {sample_ids}
        else:
            self.sample_ids = set(sample_ids if sample_ids else [])

    @property
    def enabled(self):
        return len(self.sample_ids) > 0

    def contains(self, idx):
        if self.sample_ids == {-1}:  # -1 will visualize all samples
            self.sample_ids.add(idx)
        return idx in self.sample_ids

    def visualize(self, idx, tokens, attentions):
        assert self.contains(idx), f"samples {idx} is not included in visualization samples!"
        assert attentions.ndim == 3

        self.sample_ids.discard(idx)

        output_dir = os.path.join(self.output_dir, f"sample-{idx}")
        os.makedirs(output_dir, exist_ok=True)

        # store head averaged attention maps
        text_attention_heatmap.generate(
            tokens, attentions.mean(dim=1)[None, :, :].tolist(), os.path.join(output_dir, f"average_heads.tex")
        )

        if not self.single_file:
            for ly, attns in enumerate(attentions):
                text_attention_heatmap.generate(
                    tokens, attns[None, :, :].tolist(), os.path.join(output_dir, f"layer_{ly + 1}.tex")
                )
        else:
            text_attention_heatmap.generate(
                tokens, attentions[:, :, :].tolist(), os.path.join(output_dir, f"all_layers.tex")
            )

        if self.exit_after_vis and len(self.sample_ids) == 0:
            exit()
