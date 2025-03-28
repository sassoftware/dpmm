from dpmm.pipeline import GenerativeModel, GenerativePipeline, TableBinner


class Engine(GenerativePipeline):
    model: GenerativeModel = None

    def __init__(
        self,
        # Model Params
        synth_epsilon=1,
        delta=1e-5,
        gen_kwargs={},
        # Processing Params
        binner_type="priv-tree",
        proc_epsilon=0.1,
        n_bins="auto",
        binner_kwargs={},
        disable_processing=False,
    ):

        if "delta" not in gen_kwargs and delta is not None:
            gen_kwargs["delta"] = delta

        gen = self.model(epsilon=synth_epsilon, **gen_kwargs)

        if disable_processing:
            proc = None
        else:
            proc = TableBinner(
                binner_type=binner_type,
                binner_settings=dict(
                    epsilon=proc_epsilon, n_bins=n_bins, **binner_kwargs
                ),
            )

        super().__init__(gen=gen, proc=proc)
