"""Demonstration attention extension. Not GPT-2 integrated."""


class AttentionWrapper:
    def __init__(self, native_module):
        self.native_module = native_module

    def __call__(self, *args, **kwargs):
        return self.native_module(*args, **kwargs)


class MyAttention:
    def create(
        self,
        *,
        global_layer_idx,
        native_module,
        model_config,
        options,
    ):
        if not callable(native_module):
            raise TypeError(
                "native_module must be callable"
            )

        return AttentionWrapper(native_module)
