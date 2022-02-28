import os, time
from pathlib import Path
import shutil
import numpy as np
import argparse
import chainer
from chainer import cuda
from chainer.links import VGG16Layers as VGG
from chainer.training import extensions
import chainermn
import yaml
import source.yaml_utils as yaml_utils
from gen_models.ada_generator import AdaBIGGAN, AdaSNGAN
from dis_models.patch_discriminator import PatchDiscriminator
from updater import Updater

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/default.yml")
    # parser.add_argument("--resume", "-r", type=str, default="")
    parser.add_argument("--communicator", type=str, default="hierarchical")
    parser.add_argument("--suffix", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    now = int(time.time()) * 10 + args.suffix
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    os.makedirs(f"{config.save_path}{now}", exist_ok=True)
    shutil.copy(args.config_path, f"{config.save_path}{now}/config{now}.yml")
    shutil.copy("train.py", f"{config.save_path}{now}/train.py")
    print("snapshot->", now)

    # image size
    config.image_size = config.image_sizes[config.gan_type]
    image_size = config.image_size

    if config.gan_type == "BIGGAN":
        try:
            comm = chainermn.create_communicator(args.communicator)
        except:
            comm = None
    else:
        comm = None

    device = args.gpu if comm is None else comm.intra_rank
    cuda.get_device(device).use()

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu)
        xp = cuda.cupy
    else:
        xp = np

    np.random.seed(1234)

    if config.perceptual:
        vgg = VGG().to_gpu()
    else:
        vgg = None

    layers = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2",
              "conv4_3"]


    img = xp.array(get_dataset(image_size, config))

    if comm is None or comm.rank == 0:
        perm_dataset = np.arange(len(img))
    else:
        perm_dataset = None

    if comm is not None:
        perm_dataset = chainermn.scatter_dataset(perm_dataset, comm, shuffle=True)

    batchsize = min(img.shape[0], config.batchsize[config.gan_type])
    perm_iter = chainer.iterators.SerialIterator(perm_dataset, batch_size=batchsize)

    ims = []
    datasize = len(img)

    target = img

    # Model
    if config.gan_type == "BIGGAN":
        gen = AdaBIGGAN(config, datasize, comm=comm)
    elif config.gan_type == "SNGAN":
        gen = AdaSNGAN(config, datasize, comm=comm)

    if not config.random:  # load pre-trained generator model
        chainer.serializers.load_npz(config.snapshot[config.gan_type], gen.gen)
    gen.to_gpu(device)
    gen.gen.to_gpu(device)


    random_imgs = gen.random(tmp=0.5, n=10, truncate=False)
    # get underlying array
    random_imgs = random_imgs.array


