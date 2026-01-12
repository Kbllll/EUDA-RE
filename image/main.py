import utils
from random import shuffle

if __name__ == '__main__':
    r_path = './out'

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = ''
    template_args.comment = 'base'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0,1,2,3,4,5,6,7,8,9]
    }

    base_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'eua'
    template_args.comment = 'eua'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'alpha': [0.5],
        'beta': [2],
        'r': [0.01, 0.05, 0.1, 0.15, 0.2]
    }

    eua_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'eua_supervised'
    template_args.comment = 'eua_supervised'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'alpha': [0.5],
        'beta': [2],
        'r': [0.01, 0.05, 0.1, 0.15, 0.2]
    }

    eua_supervised_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'whitenoise'
    template_args.comment = 'whitenoise'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    whitenoise_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'whitenoise2'
    template_args.comment = 'whitenoise2'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    whitenoise2_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'whitenoise3'
    template_args.comment = 'whitenoise3'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    whitenoise3_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'whitenoise4'
    template_args.comment = 'whitenoise4'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    whitenoise4_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'whitenoise5'
    template_args.comment = 'whitenoise5'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    whitenoise5_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'gan'
    template_args.comment = 'gan'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    gan_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.device = 'cuda:0'
    template_args.enhance = 'diffusion'
    template_args.comment = 'diffusion'

    combine_space = {
        'dataset_name': ['Digits', 'ORL32'],
        'epochs': [100],
        'test_size': [0.8, 0.6, 0.4, 0.2],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    diffusion_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    all_lists = (base_args_list + eua_args_list + eua_supervised_args_list + whitenoise_args_list + whitenoise2_args_list + whitenoise3_args_list + whitenoise4_args_list + whitenoise5_args_list + gan_args_list + diffusion_args_list)
    shuffle(all_lists)
    utils.parallel_run(all_lists, workers=8)

