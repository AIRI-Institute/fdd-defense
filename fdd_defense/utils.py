def weight_reset(model):
    """
    ref: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/9
    """
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()