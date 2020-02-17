# python3.7
"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.

NOTE: If you want to use W or W+ space of StyleGAN, please do not randomly
sample the latent code, since neither W nor W+ space is subject to Gaussian
distribution. Instead, please use `generate_data.py` to get the latent vectors
from W or W+ space first, and then use `--input_latent_codes_path` option to
pass in the latent vectors.
"""

import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import PIL.Image
from torchvision import transforms

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger
from utils.manipulator import linear_interpolate

from config import config

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  # parser.add_argument('-m', '--model_name', type=str, required=True,
  #                     choices=list(MODEL_POOL),
  #                     help='Name of the model for generation. (required)')
  # parser.add_argument('-o', '--output_dir', type=str, required=True,
  #                     help='Directory to save the output results. (required)')
  # parser.add_argument('-b', '--boundary_path', type=str, required=True,
  #                     help='Path to the semantic boundary. (required)')
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images for editing. This field will be '
                           'ignored if `input_latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='Z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
  parser.add_argument('--steps', type=int, default=3,
                      help='Number of steps for image editing. (default: 10 + origin_image)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(config.OUTPUT_PATH, logger_name='generate_data')

  logger.info(f'Initializing generator.')
  gan_type = MODEL_POOL[config.MODEL_NAME]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(config.MODEL_NAME, logger)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(config.MODEL_NAME, logger)
    kwargs = {'latent_space_type': args.latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

  logger.info(f'Preparing boundary.')
  if not os.path.isfile(config.BOUNDARY_PATH):
    raise ValueError(f'Boundary `{config.BOUNDARY_PATH}` does not exist!')
  boundary = np.load(config.BOUNDARY_PATH)
  #np.save(os.path.join(config.OUTPUT_PATH, 'boundary.npy'), boundary)

  logger.info(f'Preparing latent codes.')
  if os.path.isfile(args.input_latent_codes_path):
    logger.info(f'  Load latent codes from `{args.input_latent_codes_path}`.')
    latent_codes = np.load(args.input_latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
  else:
    logger.info(f'  Sample latent codes randomly.')
    latent_codes = model.complicate_sample(args.num, **kwargs)
    # 只是简单地得到适合维度的latent code，并不是真正意义上的w或者wp
  np.save(os.path.join(config.OUTPUT_PATH, 'latent_codes.npy'), latent_codes)
  total_num = latent_codes.shape[0]

  logger.info(f'Editing {total_num} samples.')

  for sample_id in tqdm(range(total_num), leave=False):
    interpolations = linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                        boundary,
                                        start_distance=args.start_distance,
                                        end_distance=args.end_distance,
                                        steps=args.steps)
    interpolation_id = 0
    canvas = PIL.Image.new('RGB', (config.RESOLUTION*args.steps, config.RESOLUTION), 'white')
    for interpolations_batch in model.get_batch_inputs(interpolations):
      if gan_type == 'pggan':
        outputs = model.easy_synthesize(interpolations_batch)
      elif gan_type == 'stylegan':
        outputs = model.easy_synthesize(interpolations_batch, **kwargs)
      for image in outputs['image']:
        # save_path = os.path.join(config.IMAGE_PATH,
        #                          f'{sample_id:03d}_{interpolation_id:03d}.jpg')
        # cv2.imwrite(save_path, image[:, :, ::-1])
        canvas.paste(transforms.ToPILImage(mode='RGB')(image),(config.RESOLUTION*interpolation_id,0))
        interpolation_id += 1
    save_path = os.path.join(config.OUTPUT_PATH, f'{sample_id:03d}.jpg')
    canvas.save(save_path)
    assert interpolation_id == args.steps
    logger.debug(f'  Finished sample {sample_id:3d}.')
  logger.info(f'Successfully edited {total_num} samples.')


if __name__ == '__main__':
  main()
