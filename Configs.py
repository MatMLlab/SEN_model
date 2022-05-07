
from absl import flags



flags.DEFINE_integer('training_epoch', 100, 'trianing epochs')
flags.DEFINE_integer('gpu', 0, 'which gpu to use')
flags.DEFINE_string('phase', 'train', 'train or test')
flags.DEFINE_integer('use_gpu', 1, 'gpu flag, 1 for GPU and 0 for CPU')
flags.DEFINE_string('job_dir', './Results', 'Parameters file.')

flags.DEFINE_string('snapshot',
                    'stacked_cpasule_autoencoders/modelsave/',
                    'Checkpoint file.')
flags.DEFINE_string('tsne_figure_name', 'tsne.png', 'Filename for the TSNE '
                    'figure. It will be saved in the checkpoint folder.')

# These two flags are necessary for model loading. Don't change them!
flags.DEFINE_string('dataset', 'mnist', 'Don\'t change!')
flags.DEFINE_string('mode', 'hse', 'Don\'t change!.')
flags.DEFINE_string('model', 'scae', 'Don\'t change!.')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('material_size', 16, 'material size.')
flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
flags.DEFINE_boolean('use_lr_schedule', True, 'Uses learning rate schedule'
                     ' if True.')

flags.DEFINE_integer('template_size', 11, 'Template size.')
flags.DEFINE_integer('n_part_caps', 16, 'Number of part capsules.')
flags.DEFINE_integer('n_part_caps_dims', 6, 'Part caps\' dimensionality.')
flags.DEFINE_integer('n_part_special_features', 16, 'Number of special '
                     'features.')

flags.DEFINE_integer('n_channels', 1, 'Number of input channels.')

flags.DEFINE_integer('n_obj_caps', 10, 'Number of object capsules.')
flags.DEFINE_integer('n_obj_caps_params', 32, 'Dimensionality of object caps '
                     'feature vector.')

flags.DEFINE_boolean('colorize_templates', False, 'Whether to infer template '
                     'color from input.')
flags.DEFINE_boolean('use_alpha_channel', False, 'Learns per-pixel mixing '
                     'proportions for every template; otherwise mixing '
                     'probabilities are constrained to have the same value as '
                     'image pixels.')

flags.DEFINE_string('template_nonlin', 'relu1', 'Nonlinearity used to normalize'
                    ' part templates.')
flags.DEFINE_string('color_nonlin', 'relu1', 'Nonlinearity used to normalize'
                    ' template color (intensity) value.')

flags.DEFINE_float('prior_within_example_sparsity_weight', 1., 'Loss weight.')
flags.DEFINE_float('prior_between_example_sparsity_weight', 1., 'Loss weight.')
flags.DEFINE_float('posterior_within_example_sparsity_weight', 10.,
                   'Loss weight.')
flags.DEFINE_float('posterior_between_example_sparsity_weight', 10.,
                   'Loss weight.')


flags.DEFINE_string('name', 'Inos', '')
flags.mark_flag_as_required('name')

flags.DEFINE_string('logdir', 'stacked_capsule_autoencoders/checkpoints/{name}',
                    'Log and checkpoint directory for the experiment.')

flags.DEFINE_float('grad_value_clip', 0., '')
flags.DEFINE_float('grad_norm_clip', 0., '')

flags.DEFINE_float('ema', .9, 'Exponential moving average weight for smoothing '
                   'reported results.')

flags.DEFINE_integer('run_updates_every', 10, '')
flags.DEFINE_boolean('global_ema_update', True, '')

flags.DEFINE_integer('max_train_steps', int(3e5), '')
flags.DEFINE_integer('snapshot_secs', 3600, '')
flags.DEFINE_integer('snapshot_steps', 0, '')
flags.DEFINE_integer('snapshots_to_keep', 5, '')
flags.DEFINE_integer('summary_steps', 500, '')

flags.DEFINE_integer('report_loss_steps', 500, '')

flags.DEFINE_boolean('plot', False, 'Produces intermediate results plots '
                     'if True.')
flags.DEFINE_integer('plot_steps', 1000, '')

flags.DEFINE_boolean('overwrite', False, 'Overwrites any existing run of the '
                     'same name if True; otherwise it tries to restore the '
                     'model if a checkpoint exists.')

flags.DEFINE_boolean('check_numerics', False, 'Adds check numerics ops.')
