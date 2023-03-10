import os
import warnings
import numpy as np
import torch as th
import random
import math
import urllib.request
import torch.distributed as dist
from torchvision import transforms, datasets

from PIL import Image, ImageFilter, ImageOps


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        #img = img.resize( (img.size[0]/8, img.size[1]/8), Image.BILINEAR )
        return img.convert('RGB')


def set_bn_eval(module):
    if isinstance(module, th.nn.modules.batchnorm._BatchNorm):
        module.eval()


class ImageFolderWithIndices(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithIndices, self).__getitem__(index)
        
        # make a new tuple that includes original and the index
        tuple_with_path = (original_tuple + (index,))
        #print(tuple_with_path)
        return tuple_with_path


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def set_lsf_env(world_size):
    ngpus_per_node = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    local_rank = int(os.environ.get('LSF_PM_XPROCID', 1)) - 1
    node_rank = int(os.environ.get('LSF_PM_XMACHID', 1)) - 1
    rank = node_rank * ngpus_per_node + local_rank
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_PORT'] = '12345'
    os.environ['MASTER_ADDR'] = os.environ.get('LSF_FROM_HOST', 'localhost')

    return rank, local_rank


class PrintMultiple(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with th.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DataAugmentation(object):
    # taken from DINO
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = list()
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        #print(crops[0])
        return crops


# taken from DINO
def cosine_scheduler_with_warmup(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    final_value = base_value if final_value is None else final_value
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_params_groups(model, args):
    if not args.no_bias_wd and args.bbone_wd is None:
        return model.parameters()
    else:
        regularized = []
        not_regularized = []
        bbone_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (name.endswith(".bias") or len(param.shape) == 1) and args.no_bias_wd:
                not_regularized.append(param)
            elif args.bbone_wd is not None and 'backbone' in name:
                bbone_regularized.append(param)
            else:
                regularized.append(param)

        param_groups = [{'params': regularized}]
        if len(not_regularized):
            param_groups.append({'params': not_regularized, 'weight_decay': 0.})
        if len(bbone_regularized):
            param_groups.append({'params': bbone_regularized, 'weight_decay': args.bbone_wd})

    return param_groups


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def imagenet_subset_samples(dataset, traindir, label_subset):
    # extract subset of training images
    subset_file = urllib.request.urlopen(
        "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/" +
        str(label_subset) + "percent.txt")
    labeled_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
    # update dataset
    dataset.samples = [(os.path.join(traindir, li.split('_')[0], li), dataset.class_to_idx[li.split('_')[0]])
                       for li in labeled_imgs]

    return dataset


class AllGather(th.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [th.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return th.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(th.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def keep_current(tensor):
    if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
        s = (tensor.shape[0] // dist.get_world_size()) * dist.get_rank()
        e = (tensor.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
        return tensor[s:e]
    return tensor


class NNQueue:
    def __init__(self, queue_len=131072, dim=128, gpu=None):
        super().__init__()
        self.queue_len = queue_len
        self.dim = dim

        self.queue = th.zeros(self.queue_len, self.dim)
        self.queue_targets = th.zeros(self.queue_len)  # only used for monitoring progress
        self.queue_indices = th.zeros(self.queue_len, dtype=th.long)  # used to avoid choosing the same sample as NN

        if th.cuda.is_available():
            self.queue = self.queue.cuda(gpu, non_blocking=True)
            self.queue_targets = self.queue_targets.cuda(gpu, non_blocking=True)
            self.queue_indices = self.queue_indices.cuda(gpu, non_blocking=True)

        self.ptr = 0
        self.full = False

    def get_nn(self, x, x_indices):
        # extract top2 in case first sample is the query sample itself which can happen
        # in the first few iterations of a new epoch
        _, q_indices = (x @ self.queue.T).topk(2)  # extract indices of queue for top2
        sample_indices = self.queue_indices[q_indices]  # extract 'global' indices of extracted samples
        indices = th.where(x_indices == sample_indices[:, 0], q_indices[:, 1], q_indices[:, 0])

        # extract values
        out = self.queue[indices]
        targets = self.queue_targets[indices]  # only used for monitoring progress, not for training
        return out, targets

    def push(self, x, x_targets, x_indices):
        x_size = x.shape[0]
        old_ptr = self.ptr
        if self.ptr + x_size <= self.queue_len:
            self.queue[self.ptr: self.ptr + x_size] = x
            self.queue_targets[self.ptr: self.ptr + x_size] = x_targets
            self.queue_indices[self.ptr: self.ptr + x_size] = x_indices
            self.ptr = (self.ptr + x_size) % self.queue_len

        else:
            self.queue[self.ptr:] = x[:self.queue_len - old_ptr]
            self.queue_targets[self.ptr:] = x_targets[:self.queue_len - old_ptr]
            self.queue_indices[self.ptr:] = x_indices[:self.queue_len - old_ptr]

            self.ptr = (self.ptr + x_size) % self.queue_len

            self.queue[:self.ptr] = x[self.queue_len - old_ptr:]
            self.queue_targets[:self.ptr] = x_targets[self.queue_len - old_ptr:]
            self.queue_indices[:self.ptr] = x_indices[self.queue_len - old_ptr:]

        if not self.full and old_ptr + x_size >= self.queue_len:
            self.full = True


def imagenet_subset(dataset, subset_file):
    # read the subset of classes to include (sorted)
    with open(subset_file, 'r') as f:
        result = f.read().splitlines()
    subdirs = [line.split(' ', 1)[0] for line in result]
    class_indices = [dataset.class_to_idx[subdir] for subdir in subdirs]
    # update dataset
    dataset.samples = [(sample_path, class_indices.index(class_index)) for sample_path, class_index in dataset.samples
                       if class_index in class_indices]
    dataset.targets = [s[1] for s in dataset.samples]

    return dataset

