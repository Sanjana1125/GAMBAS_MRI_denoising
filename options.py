import argparse


class Options:
    # Data
    data_path  = 'train_data_L_to_M/train'   # sliced from prepared_3T_47mT_pairs
    val_path   = 'train_data_L_to_M/val'      # via slice_prepared_pairs.py

    # Model
    netG       = 'gambas'
    netD       = 'basic'
    imgH       = 136
    imgW       = 152   # was 150 -- MUST be 152 to match slice_prepared_pairs.py's
                        # TARGET_W and GAMBAS's hardcoded 34x38 Mamba bottleneck grid
                        # (136/4=34, 152/4=38). 150 does not divide cleanly and will
                        # crash in MambaLayer.forward's reshape, same as before.
    input_nc   = 1
    output_nc  = 1
    ngf        = 64
    ndf        = 64
    norm       = 'instance'
    init_type  = 'normal'
    init_gain  = 0.02
    no_dropout = True
    model      = 'gambas'

    # Training
    isTrain         = True
    lr              = 2e-4
    beta1           = 0.5
    niter           = 300
    niter_decay     = 0
    batch_size      = 1
    save_epoch_freq = 50
    checkpoints_dir = 'checkpoints'
    name            = 'gambas_resnet_overfit'
    which_epoch     = 'latest'
    continue_train  = False
    epoch_count     = 1
    lr_policy       = 'lambda'

    # Loss weights
    lambda_A    = 100.0
    lambda_adv  = 0.0
    lambda_f    = 0.0
    # NOTE: these two were referenced in gambas_model.py's backward_G but
    # were never actually defined here -- would have crashed at runtime.
    # Defaults below are reasonable starting points; tune empirically.
    lambda_perc = 0.0
    lambda_ssim = 0.0

    # GAN
    no_lsgan   = False
    pool_size  = 50
    n_layers_D = 3

    # Device
    device      = 'mps'
    gpu_ids     = [0]
    num_workers = 0

    # Logging
    verbose         = False
    which_direction = 'AtoB'

    @classmethod
    def parse(cls):
        """
        Builds an Options instance, then overrides any matching attribute
        from the command line. Only attributes that already exist on the
        class can be overridden -- this is deliberate, so a typo'd flag
        (e.g. --lambda_advv) fails loudly via argparse rather than silently
        being ignored.

        Usage:
            opt = Options.parse()
            # then run as normal: python train.py --lambda_adv 0.2 --name gambas_adv02
        """
        defaults = {k: v for k, v in vars(cls).items()
                    if not k.startswith('_') and not callable(v)}

        parser = argparse.ArgumentParser()
        for key, default_val in defaults.items():
            if isinstance(default_val, bool):
                # bool flags need special handling -- argparse's type=bool
                # would treat the string "False" as truthy, which is wrong.
                parser.add_argument(f'--{key}', type=lambda x: x.lower() == 'true',
                                     default=default_val)
            elif isinstance(default_val, list):
                # gpu_ids etc. -- accept comma-separated ints, e.g. --gpu_ids 0,1
                parser.add_argument(f'--{key}', type=str, default=None)
            else:
                parser.add_argument(f'--{key}', type=type(default_val), default=default_val)

        args = parser.parse_args()

        opt = cls()
        for key in defaults:
            val = getattr(args, key)
            if isinstance(defaults[key], list) and val is not None:
                val = [int(v) for v in val.split(',') if v != '']
            elif isinstance(defaults[key], list) and val is None:
                val = defaults[key]
            setattr(opt, key, val)
        return opt