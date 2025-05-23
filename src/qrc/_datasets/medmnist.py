"""
Adapted from https://github.com/MedMNIST/MedMNIST, under Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0
"""

import hashlib
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from ._medmnist_info import DEFAULT_ROOT, HOMEPAGE, INFO


class MedMNIST:
  flag = ...

  def __init__(
    self,
    split,
    transform=None,
    target_transform=None,
    download=None,
    as_rgb=False,
    root=DEFAULT_ROOT,
    size=None,
    mmap_mode=None,
  ):
    """
    Args:

      split (string): 'train', 'val' or 'test', required
      transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
      target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
      download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default: False.
      as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
      size (int, optional): The size of the returned images. If None, use MNIST-like 28. Default: None.
      mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.
      root (string, optional): Root directory of dataset. Default: `~/.medmnist`.

    """

    # Here, `size_flag` is blank for 28 images, and `_size` for larger images, e.g., "_64".
    if (size is None) or (size == 28):
      self.size = 28
      self.size_flag = ""
    else:
      assert size in self.available_sizes
      self.size = size
      self.size_flag = f"_{size}"

    self.info = INFO[self.flag]

    if root is not None and os.path.exists(root):
      self.root = root
    else:
      raise RuntimeError(
        "Failed to setup the default `root` directory. "
        + "Please specify and create the `root` directory manually."
      )

    if download:
      self.download()

    if not os.path.exists(os.path.join(self.root, f"{self.flag}{self.size_flag}.npz")):
      if download is None:
        self.download()
      else:
        raise RuntimeError(
          "Dataset not found. "
          + " You can set `download=True` or `None` to download it"
        )

    npz_file = np.load(
      os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"),
      mmap_mode=mmap_mode,
    )

    self.split = split
    self.transform = transform
    self.target_transform = target_transform
    self.as_rgb = as_rgb

    if self.split in ["train", "val", "test"]:
      self.imgs = npz_file[f"{self.split}_images"]
      self.labels = npz_file[f"{self.split}_labels"]
    else:
      raise ValueError

  def __len__(self):
    assert self.info["n_samples"][self.split] == self.imgs.shape[0]
    return self.imgs.shape[0]

  def __repr__(self):
    """Adapted from torchvision."""
    _repr_indent = 4
    head = f"Dataset {self.__class__.__name__} of size {self.size} ({self.flag}{self.size_flag})"
    body = [f"Number of datapoints: {self.__len__()}"]
    body.append(f"Root location: {self.root}")
    body.append(f"Split: {self.split}")
    body.append(f"Task: {self.info['task']}")
    body.append(f"Number of channels: {self.info['n_channels']}")
    body.append(f"Meaning of labels: {self.info['label']}")
    body.append(f"Number of samples: {self.info['n_samples']}")
    body.append(f"Description: {self.info['description']}")
    body.append(f"License: {self.info['license']}")

    lines = [head] + [" " * _repr_indent + line for line in body]
    return "\n".join(lines)

  def download(self):
    try:
      url = self.info[f"url{self.size_flag}"]
      target_fname = Path(self.root) / f"{self.flag}{self.size_flag}.npz"
      urlretrieve(url, target_fname)

      with open(target_fname, "rb") as f:
        bs = f.read()
      md5 = hashlib.md5(bs).digest().hex()
      assert self.info[f"MD5{self.size_flag}"] == md5, f"MD5 mismatch: {md5}"

    except Exception as err:
      raise RuntimeError(
        f"""
        Automatic download failed! Please download {self.flag}{self.size_flag}.npz manually.
        1. [Optional] Check your network connection:
          Go to {HOMEPAGE} and find the Zenodo repository
        2. Download the npz file from the Zenodo repository or its Zenodo data link:
          {self.info[f"url{self.size_flag}"]}
        3. [Optional] Verify the MD5:
          {self.info[f"MD5{self.size_flag}"]}
        4. Put the npz file under your MedMNIST root folder:
          {self.root}
        """
      ) from err


class MedMNIST2D(MedMNIST):
  available_sizes = [28, 64, 128, 224]

  def __getitem__(self, index):
    """
    return: (without transform/target_transofrm)
      img: PIL.Image
      target: np.array of `L` (L=1 for single-label)
    """
    img, target = self.imgs[index], self.labels[index].astype(int)
    img = Image.fromarray(img)

    if self.as_rgb:
      img = img.convert("RGB")

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class MedMNIST3D(MedMNIST):
  available_sizes = [28, 64]

  def __getitem__(self, index):
    """
    return: (without transform/target_transofrm)
      img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
      target: np.array of `L` (L=1 for single-label)
    """
    img, target = self.imgs[index], self.labels[index].astype(int)

    img = np.stack([img / 255.0] * (3 if self.as_rgb else 1), axis=0)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class PathMNIST(MedMNIST2D):
  flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
  flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
  flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
  flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
  flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
  flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
  flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
  flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
  flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
  flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
  flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
  flag = "organsmnist"


class OrganMNIST3D(MedMNIST3D):
  flag = "organmnist3d"


class NoduleMNIST3D(MedMNIST3D):
  flag = "nodulemnist3d"


class AdrenalMNIST3D(MedMNIST3D):
  flag = "adrenalmnist3d"


class FractureMNIST3D(MedMNIST3D):
  flag = "fracturemnist3d"


class VesselMNIST3D(MedMNIST3D):
  flag = "vesselmnist3d"


class SynapseMNIST3D(MedMNIST3D):
  flag = "synapsemnist3d"


# backward-compatible aliases
OrganMNISTAxial = OrganAMNIST
OrganMNISTCoronal = OrganCMNIST
OrganMNISTSagittal = OrganSMNIST
