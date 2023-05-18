import os
import glob


def main():
    folder = './data/DIV2K/valid_LR'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    i=860
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = str(i) + ".png"
        os.rename(img_path, new_path)
        i = i+1


if __name__ == "__main__":
    main()