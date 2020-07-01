def create_model(opt):
    model = None
    if opt.model == 'age_cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .age_cycle_gan_model import AgeCycleGANModel
        model = AgeCycleGANModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
