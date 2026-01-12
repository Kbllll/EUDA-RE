import torch

from eua import EvidentialCMeans, Degranulatuin
from whitenoise import augment_with_white_noise
from gan import augment_with_valid_labels
from diffusion import augment_with_diffusion


class Enhancer:
    def __init__(self, args):
        self.args = args

    def generate(self, train_data):
        if self.args.enhance == 'eua':
            x, y = train_data.get_raw()
            # 转换为torch tensor
            x = torch.tensor(x, dtype=torch.float32).to(self.args.device)

            # 粒化并plusu
            gra = EvidentialCMeans(self.args)
            u_plus, v_extend = gra.plus_uncertainty(x)
            # 解粒
            deg = Degranulatuin(self.args)
            x_ex = deg.anti_point(x, u_plus, v_extend)
            return x_ex, y

        if self.args.enhance == 'eua_supervised':
            x, y = train_data.get_raw()
            # 转换为torch tensor
            x = torch.tensor(x, dtype=torch.float32).to(self.args.device)

            # 粒化并plusu
            gra = EvidentialCMeans(self.args)
            u_plus, v_extend = gra.plus_uncertainty(x, y)
            # 解粒
            deg = Degranulatuin(self.args)
            x_ex = deg.anti_point(x, u_plus, v_extend)
            return x_ex, y

        if self.args.enhance == 'whitenoise':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_white_noise(x, y)

            return ex_x, ex_y

        if self.args.enhance == 'whitenoise2':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_white_noise(x, y, n_aug_per_sample=1, sigma=0.05)

            return ex_x, ex_y

        if self.args.enhance == 'whitenoise3':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_white_noise(x, y, n_aug_per_sample=1, sigma=0.1)

            return ex_x, ex_y

        if self.args.enhance == 'whitenoise4':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_white_noise(x, y, n_aug_per_sample=1, sigma=0.15)

            return ex_x, ex_y

        if self.args.enhance == 'whitenoise5':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_white_noise(x, y, n_aug_per_sample=1, sigma=0.2)

            return ex_x, ex_y

        if self.args.enhance == 'gan':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_valid_labels(x, y, img_size=int(x.shape[1] ** 0.5), device=self.args.device)

            return ex_x, ex_y

        if self.args.enhance == 'diffusion':
            x, y = train_data.get_raw()

            ex_x, ex_y = augment_with_diffusion(x, y, img_size=int(x.shape[1] ** 0.5), device=self.args.device)

            return ex_x, ex_y

        raise Exception('Unknown enhancement')
