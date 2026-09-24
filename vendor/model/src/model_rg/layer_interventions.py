"""Reversible interventions on explicitly selected native decoder operators."""
from contextlib import contextmanager
from model_rg.inference_interventions import fixed_operators


def selected_layers(model, layers):
    chosen=tuple(layers)
    if not chosen or len(set(chosen))!=len(chosen) or any(type(i) is not int or i<0 or i>=len(model.model.decoder.dec_layers) for i in chosen):
        raise ValueError('A nonempty set of valid native decoder indices is required')
    return set(chosen)


@contextmanager
def fixed_layer_operators(model, cache, layers):
    """Use native external operators only in the declared decoders, with no KV cache.

    The caller must use use_cache=False. Unselected modules keep their
    original native generator branches; selected modules use the author's
    existing external-operator implementation. All original flags and
    buffers are restored by the enclosing qualified context manager.
    """
    chosen=selected_layers(model,layers)
    originals=[(decoder.mha1.custom_G_type,decoder.mha1.plgatt_layer.custom_G_type)
               for decoder in model.model.decoder.dec_layers]
    with fixed_operators(model,cache):
        for index,decoder in enumerate(model.model.decoder.dec_layers):
            if index not in chosen:
                decoder.mha1.custom_G_type,decoder.mha1.plgatt_layer.custom_G_type=originals[index]
        yield


@contextmanager
def projected_layer_rows(model, layers):
    """Project only the selected emitted metrics onto their common-row subspace."""
    chosen=selected_layers(model,layers);handles=[]
    def project(module,arguments):
        inputs,*rest=arguments
        query,key,value,matrix,mask=inputs
        reduced=matrix.mean(-2,keepdim=True).expand_as(matrix).contiguous()
        return ((query,key,value,reduced,mask),*rest)
    try:
        for index,decoder in enumerate(model.model.decoder.dec_layers):
            if index in chosen:handles.append(decoder.mha1.plgatt_layer.register_forward_pre_hook(project))
        yield
    finally:
        for handle in handles:handle.remove()
