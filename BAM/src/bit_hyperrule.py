
known_dataset_sizes = {
  # 'cifar10': (32, 32),
  # 'cifar100': (32, 32),
  # 'oxford_iiit_pet': (224, 224),
  # 'oxford_flowers102': (224, 224),
  # 'imagenet2012': (224, 224),
  'custom': (512, 512),
}


def get_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)


def get_resolution_from_dataset(dataset):
  if dataset not in known_dataset_sizes:
    raise ValueError(f"Unsupported dataset {dataset}. Add your own here :)")
  return get_resolution(known_dataset_sizes[dataset])


def get_mixup(dataset_size):
  return 0.0 if dataset_size < 20_000 else 0.1


def get_schedule(dataset_size):
  if dataset_size < 20_000:
    return [100, 200, 300, 400, 500]
#    return [400, 500]
#    return [200, 300]
#     return [100, 200] # , 300, 400, 500] 
  elif dataset_size < 500_000:
    return [500, 3000, 6000, 9000, 10_000]
  else:
    return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
  """Returns learning-rate for `step` or None at the end."""
  supports = get_schedule(dataset_size)
  # Linear warmup
  if step < supports[0]:
    return base_lr * step / supports[0]
  # End of training
  elif step >= supports[-1]:
    return None
  # Staircase decays by factor of 10
  else:
    for s in supports[1:]:
      if s < step:
        base_lr /= 10
    return base_lr
