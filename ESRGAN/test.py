from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from modules.models import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)


flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    # evaluation
    if FLAGS.img_path:
        print("[*] Processing on single image {}".format(FLAGS.img_path))
        raw_img = cv2.imread(FLAGS.img_path)
        #lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale'])
        low_resolution_shape = (32, 32, 3) # Shape of low-resolution and high-resolution images
        high_resolution_shape = (128, 128, 3)
        hr_img = resize(raw_img, high_resolution_shape) # Resize the image
        lr_img = resize(raw_img, low_resolution_shape)

        sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
        #bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)
        bic_img = lr_img

        str_format = "[{}] PSNR/SSIM: Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
        print(str_format.format(
            os.path.basename(FLAGS.img_path),
            calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
            calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
            calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
            calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
        result_img_path = './Bic_SR_HR_' + os.path.basename(FLAGS.img_path)
        print("[*] write the result image {}".format(result_img_path))
        results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
        cv2.imwrite(result_img_path, results_img)
    else:
        print("[*] Processing on various datasets, and writing results")
        results_path = './results/' + cfg['sub_name'] + '/'

        text_file = open("Results.txt", "w")
        psnr_val = []
        ssim_val = []

        for key, path in cfg['test_dataset'].items():
            print("'{}' form {}\n  PSNR/SSIM".format(key, path))
            dataset_name = key.replace('_path', '')
            pathlib.Path(results_path + dataset_name).mkdir(
                parents=True, exist_ok=True)

            psnr_set_val = []
            ssim_set_val = []

            for img_name in os.listdir(path):
                raw_img = cv2.imread(os.path.join(path, img_name))
                lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale'])
                sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
                bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)
                str_format = "  [{}] Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
                print(str_format.format(
                    img_name + ' ' * max(0, 20 - len(img_name)),
                    calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                    calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                    calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                    calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
                psnr_val.append(calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)))
                ssim_val.append(calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)))
                psnr_set_val.append(calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)))
                ssim_set_val.append(calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)))
                result_image = "IMAGE = " + img_name + " \t     |||     PSNR = " + str(calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))) + " \t     |||     SSIM = " + str(calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))) + "\n"
                text_file.write(result_image)
                result_img_path = os.path.join(
                    results_path + dataset_name, 'Bic_SR_HR_' + img_name)
                results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
                cv2.imwrite(result_img_path, results_img)

            final_set_result = "   MEAN SET PSNR = " + str(np.mean(psnr_set_val)) + "   :::   MEAN SET SSIM = " + str(np.mean(ssim_set_val))
            text_file.write(final_set_result + "\n")
            text_file.write("-------------------\n")

        print("[*] write the visual results in {}".format(results_path))

    final_result = "   MEAN PSNR = " + str(np.mean(psnr_val)) + "   :::   MEAN SSIM = " + str(np.mean(ssim_val))
    text_file.write(final_result)
    text_file.close()    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
