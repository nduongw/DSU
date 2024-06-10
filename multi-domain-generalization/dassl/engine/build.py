from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry('TRAINER')


def build_trainer(cfg, args):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print('Loading trainer: {}'.format(cfg.TRAINER.NAME))
    
    import pdb; pdb.set_trace()
    print(f'{TRAINER_REGISTRY.get(cfg.TRAINER.NAME)}')
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg, args)

