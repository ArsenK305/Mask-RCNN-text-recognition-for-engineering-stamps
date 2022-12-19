import PIL.Image
from PIL import ImageEnhance, Image
import numpy
import PySimpleGUI as sg
import os
from io import BytesIO
import skimage
import time
from pathlib import Path
import matplotlib.pyplot as plt
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
import json
import scipy.ndimage
import regex as re
from datetime import datetime

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.visualize import image_mask_and_boxes, instances_to_images, create_image_boxed
from mrcnn.use_tesseract import image_to_string, image_to_txt, image_to_df
from mrcnn.image_processing import pre_mask2, pre_mask1

ROOT_DIR = os.path.abspath("../../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CustomConfig2(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 18  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95


class CustomConfigBlueprint(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "BLUEPRINT"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Blueprint 2.0 (06.10.22) dataset from Model_creating_process\data\Blueprint2.0\
############################################################


class CustomConfigBlueprint2(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "BLUEPRINT2"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95


class CustomConfigStamp(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "stamp"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 25  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################


class CustomConfigStamp2(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "stamp"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 34  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6


class InferenceConfigStamp2(CustomConfigStamp2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfig(CustomConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfig2(CustomConfig2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfigBlueprint(CustomConfigBlueprint):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfigBlueprint2(CustomConfigBlueprint2):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class InferenceConfigStamp(CustomConfigStamp):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    raise

            # delete old key
            del d[key]
    return d


def is_valid_path_weights(filepath):
    if filepath and Path(filepath).exists():
        return True
    else:
        sg.popup_error("No filepath for weights")
        return False


def is_valid_path_image(filepath):
    if filepath and Path(filepath).exists():
        return True
    else:
        sg.popup_error("No filepath")
        return False


def splash(model, class_names, arr_image):
    if arr_image.shape[-1] == 4:
        arr_image = arr_image[..., :3]
    r = model.detect([arr_image], verbose=0)[0]
    image = image_mask_and_boxes(arr_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return image


def two_mask_splash(model1, model2, class_names1, class_names2, image_path):
    print("Runnin on {}".format(image_path))
    image0 = skimage.io.imread(image_path)
    image0 = pre_mask2(image0)
    if image0.shape[-1] == 4:
        image0 = image0[..., :3]
    # Detect using Mask1
    r0 = model2.detect([image0], verbose=0)[0]
    image1 = image_mask_and_boxes(image0, r0['rois'], r0['masks'], r0['class_ids'], class_names2, r0['scores'])
    skimage.io.imshow(image1)
    plt.show()
    # image0 = pre_mask1(image0)
    dict_cropped = instances_to_images(image0, r0['rois'], r0['class_ids'], list_accepted_ids=[1, 2])
    list_img = []
    for id in dict_cropped:
        if isinstance(dict_cropped[id], list):
            for i in dict_cropped[id]:
                image_result = splash(model1, class_names1, i)
                list_img.append(image_result)
        else:
            image_result = splash(model1, class_names1, dict_cropped[id])
            list_img.append(image_result)
    return list_img


def crop_instances(image, boxes, class_ids, what_to_crop):
    white_big = skimage.io.imread("white.jpg")
    print(white_big.shape)
    white_big_rgb = skimage.color.gray2rgb(white_big)
    n_instances = boxes.shape[0]
    for i in range(n_instances):
        if class_ids[i] in what_to_crop:
            y = boxes[i][0]
            x = boxes[i][1]
            height = boxes[i][2]
            width = boxes[i][3]
            # white_small = white_big[0: width - x, 0: height - y]
            image[y:height, x:width] = white_big_rgb[0:height - y, 0: width - x]
    return image, what_to_crop


def generate_data_stamp(model, folderpath):
    for i in os.listdir(folderpath):
        if i.endswith(".pdf"):
            image = get_image_for_blueprint(folderpath + "\\" + i)
            # image = Image.open(folderpath + '\\' + i)
            # image = skimage.io.imread(folderpath + "\\" + i)
            print(image.shape)
            r = model.detect([image], verbose=0)[0]
            dict_images = instances_to_images(image, r['rois'], r['class_ids'], names=class_names_model2,
                                              list_accepted_ids=[1, 2], stamp=False)

            # image = pre_mask2(image)
            # if image.shape[-1] == 4:
            #     image = image[..., :3]
            # r = model.detect([image], verbose=0)[0]
            # dict_images = instances_to_images(image, r['rois'], r['class_ids'], list_accepted_ids=[1,2], stamp = True)
            counter = 1
            for id in dict_images:
                if isinstance(dict_images[id], list):
                    for im in dict_images[id]:
                        # im = pre_mask1(im)
                        im = Image.fromarray(im)
                        width, height = im.size
                        print(im.size)
                        print(type(im))
                        # if width > height *2:
                        #     image = im.resize((874, int(height / width * 874)))
                        # else:
                        # image = im.resize((int(width / height * 874), 874))
                        im.save(folderpath + "\\stamp\\" + str(counter) + i.replace("pdf", "jpg"), 'JPEG')
                        counter += 1
                else:
                    im = Image.fromarray(dict_images[id])
                    width, height = im.size
                    print(im.size)
                    print(type(im))
                    # if width > height * 2:
                    #     image = im.resize((874, int(height / width * 874)))
                    # else:
                    # image = im.resize((int(width / height * 874), 874))
                    im.save(
                        folderpath + "\\stamp\\" + str(counter) + i.replace("pdf", "jpg"), 'JPEG')
                    counter += 1
    sg.popup_notify("Completed!")


def generate_data_tesseract(model1, model2, folderpath):
    fp = "C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\tesseract\\common\\"
    for file in os.listdir(folderpath):
        if file.endswith(".pdf"):
            image = get_image_for_blueprint(folderpath + "\\" + file)
            r = model2.detect([image], verbose=0)[0]
            dict_images = instances_to_images(image, r['rois'], r['class_ids'], list_accepted_ids=[1, 2], stamp=False)

            counter = 0
            for id in dict_images:

                if isinstance(dict_images[id], list):

                    for im in dict_images[id]:
                        image_to_txt(im, fp + str(counter))
                        skimage.io.imsave(fp + str(counter) + file, im)
                        counter += 1
                else:
                    image_to_txt(dict_images)
                    skimage.io.imsave(
                        fp + str(counter) + file,
                        dict_images[id]
                    )
                    counter += 1


def generate_data_tesseract_1model(model1, folderpath):
    fp = ""
    counter = 0
    for file in os.listdir(folderpath):
        if file.endswith(".jpg"):
            image = skimage.io.imread(folderpath + "\\" + file)
            if image.shape[-1] == 4:
                image = image[..., :3]
            r1 = model1.detect([image], verbose=0)[0]
            dict_images_stamp = instances_to_images(image, r1['rois'], r1['class_ids'], names=class_names_model1,
                                                    list_accepted_ids=[1, 2], stamp=True)

            counter = 0
            for id in dict_images_stamp:
                if isinstance(dict_images_stamp[id], list):
                    for im in dict_images_stamp[id]:
                        image_to_txt(im, fp + id + str(counter))
                        im = Image.fromarray(im)
                        im.save(fp + id + str(counter) + file + "jpg", "JPEG")
                        counter += 1
                else:
                    image_to_txt(dict_images_stamp[id], fp + id + str(counter))
                    im = Image.fromarray(dict_images_stamp[id])
                    im.save(fp + id + str(counter) + file + "jpg", "JPEG")
                    counter += 1


def generate_data_tesseract_from_pdf_foler(model1, model2, folder_path):
    print(folder_path)
    fp = folder_path + "/True"
    print(fp)
    counter = 0
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            image = get_image_for_blueprint(folder_path + "\\" + file)
            r = model2.detect([image], verbose=0)[0]
            dict_images = instances_to_images(image, r['rois'], r['class_ids'], names=class_names_model2,
                                              list_accepted_ids=[1, 2], stamp=False)
            for id in dict_images:
                r1 = model1.detect([dict_images[id]], verbose=0)[0]
                dict_images_stamp = instances_to_images(dict_images[id], r1['rois'], r1['class_ids'],
                                                        names=class_names_model1, list_accepted_ids=[True], stamp=True)

                for id in dict_images_stamp:
                    # if id not in ["Company_name", "Stamp", "Label_title_h", "Label_title_v", "Label_Proj_no_h",
                    #               "Label_Proj_no_v", "Label_dr_no_h", "Label_dr_no_v", "Label_rev_h", "Label_rev_v",
                    #               "Label_scale_h", "Label_scale_v", "Label_date_h", "Label_date_v", "Label_by",
                    #               "Label_chk", "Label_eng", "Label_supv", "Label_mgr", "Label_oper", "Role_input"]:
                    if id in ["Proj_no_h", "Proj_no_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            for im in dict_images_stamp[id]:
                                image_to_txt(im, fp + id + str(counter))
                                im = Image.fromarray(im)
                                im.save(fp + "/" + id + str(counter) + ".jpg", "JPEG")
                                counter += 1
                        else:
                            image_to_txt(dict_images_stamp[id], fp + "/" + id + str(counter))
                            im = Image.fromarray(dict_images_stamp[id])
                            im.save(fp + "/" + id + str(counter) + ".jpg", "JPEG")
                            counter += 1


def generate_data_for_Mask2(pdf_folder_path):
    counter = 300
    for file in os.listdir(path=pdf_folder_path):
        if file.endswith('.pdf'):
            filepath = pdf_folder_path + '\\' + file
            pages = convert_from_path(os.path.abspath(filepath), 300)
            for page in pages:
                width, height = page.size
                if width > height:
                    # print(page.size)  # tuple : (width, height)
                    image = page.resize((4096, int(height / width * 4096)))
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(2)  # Sharpness
                    enhancer = ImageEnhance.Color(image)
                    image = enhancer.enhance(0)  # black and white
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2)  # Contrast
                    image = numpy.asarray(image)  # array
                    image_arr8 = image.astype(numpy.uint8)
                    image = Image.fromarray(image_arr8)
                    imagepath = pdf_folder_path + "\\images\\" + str(counter) + '.jpg'
                    print(type(image_arr8))
                    print(type(image))
                    image.save(imagepath, 'JPEG', quality=100, dpi=(300, 300))
                    counter += 1


def full_tesseract(model, model2, imagepath):
    image = skimage.io.imread(imagepath)
    image = pre_mask2(image)
    if image.shape[-1] == 4:
        image = image[..., :3]

    r0 = model2.detect([image], verbose=0)[0]
    print(r0['class_ids'])
    n_instances = r0['rois'].shape[0]
    for i in range(n_instances):
        if not np.any(r0['rois'][i]) or not r0['class_ids'][i] in [1, 2]:
            continue
        box = r0['rois'][i]
        boxed_image = create_image_boxed(image, box)
    # boxed_image = pre_mask1(boxed_image)
    # skimage.io.imshow(boxed_image)
    # plt.show()
    # print(type(image))
    # print(type(boxed_image))
    # # skimage.io.imshow(boxed_image)
    # # plt.show()
    # print(boxed_image.shape)

    r = model.detect([boxed_image], verbose=0)[0]
    image_pure = crop_instances(boxed_image, r['rois'], r['class_ids'], [11, 12, 13, 14, 15, 16])
    # skimage.io.imshow(image_pure)
    # plt.show()
    print(r['class_ids'])
    dict_im = instances_to_images(image_pure, r['rois'], r['class_ids'],
                                  list_accepted_ids=[3, 4, 5, 6, 7, 8, 9, 10, 24], stamp=True)

    dict_text = {}
    for id in dict_im:
        if isinstance(dict_im[id], list):
            list_text = []
            for i in dict_im[id]:
                text = image_to_string(i, lang="eng1.3+rus1.3", config=r'--oem 3 --psm 6 -c page_separator=""')
                list_text.append(text)
            dict_text[int(id)] = list_text
        else:
            text = image_to_string(dict_im[id], lang="eng1.3+rus1.3", config=r'--oem 3 --psm 6 -c page_separator=""')
            dict_text[int(id)] = text

    print(dict_text)
    with open("sample.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4)
    with open("sample_no_ascii.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4, ensure_ascii=False)


def get_image_for_blueprint(path, dpi):
    """Create image for full blueprint from path
    path: path to pdf file
    """
    images = convert_from_bytes(open(path, 'rb').read(), dpi=dpi)  # list PIL images
    for i in images:
        width, height = i.size
        if width < height:
            continue
        else:
            print(i.size)  # tuple : (width, height)
            image = i.resize((4096, int(height / width * 4096)))
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2)  # Sharpness
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0)  # black and white
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)  # Contrast
            image = numpy.asarray(image)  # array
            image = image.astype(numpy.uint8)
            return image


def full_tesseract_pdf(model, model2, path):
    image = get_image_for_blueprint(path)

    r0 = model2.detect([image], verbose=0)[0]
    print(r0['class_ids'])
    n_instances = r0['rois'].shape[0]
    for i in range(n_instances):
        try:
            boxed_image
            continue
        except NameError:
            if not np.any(r0['rois'][i]) or not r0['class_ids'][i] in [1, 2]:
                continue
            print(i)
            if i == 2:
                if (1 in r0['class_ids'] and 2 in r0['class_ids']):
                    i = np.where(r0['class_ids'] == 2)[0][0]
            print(i)
            box = r0['rois'][i]
            boxed_image = create_image_boxed(image, box)

    boxed_image = Image.fromarray(boxed_image)
    boxed_image.show()
    width, height = boxed_image.size
    boxed_image = boxed_image.resize((int(width / height * 874), 874))
    boxed_image = numpy.asarray(boxed_image)  # array
    boxed_image = boxed_image.astype(numpy.uint8)

    r = model.detect([boxed_image], verbose=0)[0]
    image_pure = crop_instances(boxed_image, r['rois'], r['class_ids'],
                                [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    image_pure = Image.fromarray(image_pure)
    width, height = image_pure.size
    # image_pure = image_pure.resize((int(width / height * 874), 874))
    image_pure.show()
    image_pure = numpy.asarray(image_pure)  # array
    image_pure = image_pure.astype(numpy.uint8)
    print(r['class_ids'])
    dict_im = instances_to_images(image_pure, r['rois'], r['class_ids'], names=class_names_model1,
                                  list_accepted_ids=[3, 4, 5, 6, 7, 8, 9, 10, 24, 25], stamp=True)

    dict_text = {}
    for id in dict_im:
        if isinstance(dict_im[id], list):
            list_text = []
            for i in dict_im[id]:
                # im = Image.fromarray(i)
                # width, height = im.size
                # print(type(im))
                # # if width > height * 2:
                # #     image = im.resize((874, int(height / width * 874)))
                # # else:
                # image = im.resize((int(width / height * 874), 874))
                text = image_to_string(i, lang="eng1.3+rus1.3", config=r'--oem 3 --psm 6 -c page_separator=""')
                list_text.append(text)
            dict_text[id] = list_text
        else:
            # im = Image.fromarray(dict_im[id])
            # width, height = im.size
            # print(type(im))
            # # if width > height * 2:
            # #     image = im.resize((874, int(height / width * 874)))
            # # else:
            # image = im.resize((int(width / height * 874), 874))
            text = image_to_string(dict_im[id], lang="eng1.3+rus1.3", config=r'--oem 3 --psm 6 -c page_separator=""')
            dict_text[id] = text

    print(dict_text)
    with open("sample.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4)
    with open("sample_no_ascii.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4, ensure_ascii=False)


def datetime_extract(image):
    patterns = [
        re.compile(r"^[0-9]+[/.\\-][0-9]+[/.\\-][0-9]+$", re.IGNORECASE),
        re.compile(r"^[0-9]+[/.\\-][0-9]+[/.\\-][0-9]+$", re.IGNORECASE)

    ]
    patterns2 = [
        re.compile(r"^[0-9]+[/.\\-][0-9]+$", re.IGNORECASE), re.compile(r"^[0-9]{2}[/.\\-][0-9]+$", re.IGNORECASE)
    ]
    # date_re = re.match([0-9]{2}[-\\. ]{1,2}[0-9]{2}[-\\. ]{1,2}(19|20)[0-9]{2})
    dt = None
    content = image_to_string(image, lang="eng1.9", config=r'--oem 3 --psm 7 -c page_separator=""')
    print(content)
    for pattern in patterns:
        print(pattern)
        try:
            dt = re.search(pattern, content).group()
            print(dt)
            break
        except AttributeError:
            continue
    if dt is None:
        for pattern in patterns2:
            print(pattern)
            try:
                dt = re.search(pattern, content).group()
                print(dt)
                if pattern == re.compile(r"^[0-9]{2}[/.\\-][0-9]+$", re.IGNORECASE):
                    dt = dt[0:5] + r"/" + dt[6::]
                    break
                elif pattern == re.compile(r"^[0-9]+[/.\\-][0-9]+$", re.IGNORECASE):
                    dt = dt[0:2] + r"/" + dt[3::]
                    break
            except AttributeError:
                continue
        if dt is None:
            print("No match")
            return content  # Исправлять дт в этом месте два кейса 21117/22 и 21/17121
    return dt


def replace_numbers(text):
    """
    Replaces all numbers instead of characters from 'text'.

    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'[a-zA-Z]+', text)
    result_number = ''.join(list_of_numbers)
    return result_number


def check_O(text):
    if "0" in text:
        return text.replace("0", "O")
    else:
        return text


def replace_code(text):
    for iter in re.finditer(re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)"), text):
        result_text = ''.join(iter.group())
        return result_text
    return text


def list_duplicates_of(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def array_indexes_of_duplicates_of(arr, what_include=None):  # Возвращает индексы values(повторяющихся) в arr
    if what_include is None:
        what_include = [3, 4, 7, 8, 11, 12, 15, 16, 23, 24]
    arr_copy = [id for id in arr if id in what_include]
    values_arr, counts_arr = np.unique(arr_copy,
                                       return_counts=True)  # values = np.array([ВСЕ УНИКАЛЬНЫЕ ЗНАЧЕНИЯ])       ounts_arr = np.arr([кол-во повторений])
    list_duplicates = []
    for i in range(values_arr.shape[0]):
        if counts_arr[i] > 1:
            list_duplicates.append(values_arr[i])
    return list_duplicates  # Возвращаем значения id классов которые повторяются e.g. [3, 7, 11] - 3 лейбла повторяются


def filter_array_of_array(arr_class_ids, arr_conf, arr_boxes, arr_masks, ids_list):
    list_delete = []
    for id in ids_list:  # Для каждого айди
        indexes_list = np.where(arr_class_ids == id)[0]  # Лист индексов повторяющихся айди
        list_conf = []
        for index in indexes_list:
            list_conf.append(arr_conf[index])  # Грузим лист всех конфинесов повторяющихся айди
        list_conf.remove(max(list_conf))
        if list_conf is not False:
            for conf in list_conf:
                ind_delete = np.where(arr_conf == conf)[0][0]
                list_delete.append(ind_delete)
        else:
            break
    list_delete.sort(reverse=True)
    for i in list_delete:
        arr_conf = np.delete(arr_conf, i)
        arr_class_ids = np.delete(arr_class_ids, i)
        arr_boxes = np.delete(arr_boxes, i, axis=0)
        arr_masks = np.delete(arr_masks, i, axis=2)
    return arr_conf, arr_boxes, arr_class_ids, arr_masks


def is_biggest(list, i):
    if i == max(list):
        return list.index(i)
    else:
        return None


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def full_tesseract_pdf_new(model1, model2, path, lang_comb, langeng, langrus):
    print("STARTED NEW FILE HERE \n !!!==================================================================!!!")
    time_im_b = time.time()
    image = get_image_for_blueprint(path, dpi=300)
    time_im_f = time.time() - time_im_b
    time_model2_detect_b = time.time()
    print("--- %s seconds for image from pdf conversion ---" % (time_im_f))
    r = model2.detect([image], verbose=0)[0]
    # for i, _ in enumerate(r['class_ids']):        #Для чтения всех коэфоф уверенности
    if 1 in r['class_ids'] and 2 in r['class_ids']:  # Оба штампа найдены
        # Выбираем тот в котором больше уверенности
        if r['scores'][np.where(r['class_ids'] == 1)[0][0]] > r['scores'][np.where(r['class_ids'] == 2)[0][0]]:
            accept = [1]
        else:
            accept = [2]
    else:
        accept = [1, 2]
    time_model2_detect_f = time.time() - time_model2_detect_b
    print("--- %s seconds for detection blueprint ---" % (time_model2_detect_f))
    what_cropped = ["Stamp", "Company_name", "Label_title_h", "Label_title_v", "Label_Proj_no_h",
                    "Label_Proj_no_v", "Label_dr_no_h", "Label_dr_no_v", "Label_rev_h", "Label_rev_v",
                    "Label_scale_h", "Label_scale_v", "Label_date_h", "Label_date_v"]
    dict_images = instances_to_images(image, r['rois'], r['class_ids'], names=class_names_model2,
                                      list_accepted_ids=accept, stamp=False)

    dict_text = {}
    # if np.any(dict_images["DRAWING NO"]): drawing_no_im = dict_images["DRAWING NO"] rot_img = scipy.ndimage.rotate(
    # drawing_no_im, -90) dict_text["DRAWING_NO"] = image_to_string(rot_img, lang = 'eng1.9', config=r'--oem 3 --psm
    # 7 -c page_separator=""')

    time_instances_to_im_f = time.time() - time_model2_detect_f
    print("--- %s seconds for instances to images blueprint ---" % (time_instances_to_im_f))
    try:
        for id in dict_images:
            time_loop0_b = time.time()
            r1 = model1.detect([dict_images[id]], verbose=0)[0]  # ПОЛУЧАЕМ КОРДЫ И ИМЕНА
            list_dups = array_indexes_of_duplicates_of(r1['class_ids'])  # r['class_ids'] преобразуем в лист
            r1['scores'], r1['rois'], r1['class_ids'], r1['masks'] = filter_array_of_array(r1['class_ids'],
                                                                                           r1['scores'], r1['rois'],
                                                                                           r1['masks'], list_dups)

            time_model1_detect_f = time.time() - time_loop0_b
            print("--- %s seconds for second detection stamp ---" % (time_model1_detect_f))
            stamp_image_copy = dict_images[id].copy()
            image_pure, rfederaciya = crop_instances(stamp_image_copy, r1['rois'], r1['class_ids'],
                                                     [1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 23, 24])

            # rfederaciya = what_to_crop
            im = Image.fromarray(image_pure)
            # im.show()
            dict_images_stamp = instances_to_images(image_pure, r1['rois'], r1['class_ids'], names=class_names_model1,
                                                    list_accepted_ids=[5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 27,
                                                                       28,
                                                                       29, 30, 31, 32, 33, 34], stamp=True)
            for id in dict_images_stamp:
                if id in what_cropped:
                    continue
                else:
                    if id == "Date_v" or id == "Date_h":
                        if isinstance(dict_images_stamp[id], list):
                            text = datetime_extract(dict_images_stamp[id][0])
                        else:
                            text = datetime_extract(dict_images_stamp[id])
                        dict_text[id] = text
                    elif id in ["Label_oper", "Label_eng", "Label_mgr", "Label_chk", "Label_by", "Label_supv"]:
                        try:
                            if dict_images_stamp[id]["Coordinates"] is None:
                                dict_text[id] = "Empty"
                            else:
                                text = image_to_string(dict_images_stamp[id]["Coordinates"], lang="eng1.9to1.10_3",
                                                       config=r'--oem 3 --psm 7 -c page_separator=""')
                                print("ОРИГ ЛЭЙБЛА РОЛИ: " + text)
                                text = check_O(text)
                                text = replace_numbers(text)
                                dict_text[id] = text

                        except TypeError:
                            if isinstance(dict_images_stamp[id], list):
                                if dict_images_stamp[id][0]["Coordinates"] is None:
                                    dict_text[id] = "Empty"
                                else:
                                    im = Image.fromarray(dict_images_stamp[id][0]["Coordinates"])
                                    dict_text[id] = replace_numbers(check_O(image_to_string(im, lang="eng1.9to1.10_3",
                                                                                            config=r'--oem 3 --psm 7 -c '
                                                                                                   r'page_separator=""')))
                    elif id == "Role_input":
                        pass
                    elif id in ["REV_h",
                                "REV_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="eng1.4to_proj_no3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.4to_proj_no3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        text = text.replace("O", "0")
                        dict_text[id] = text
                    elif id in ["Scale_v", "Scale_h"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="eng1.9+rus1.7",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.9+rus1.7",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        if text == "":
                            text = "-"
                        elif text is None:
                            text = "-"
                        dict_text[id] = text
                    elif id in ["Project_name_h", "Project_name_v"]:
                        # if isinstance(dict_images_stamp[id], list):
                        #     text = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_5+rus1.7to1.9_4",
                        #                            config=r'--oem 3 --psm 6 -c page_separator=""')
                        # else:
                        #     Image.fromarray(dict_images_stamp[id]).show()
                        #     text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_5+rus1.7to1.9_4",
                        #                            config=r'--oem 3 --psm 6 -c page_separator=""')
                        df = image_to_df(dict_images_stamp[id], section="text", lang=langeng, config=r'--oem 3 --psm 6 -c page_separator=""')       #Прокручиваем всё на англ языке
                        a = 0
                        sum_lines = 0
                        line_qt = df.max()['line_num']
                        if df.max()['par_num'] > 1:
                            for i in range(df.max()['par_num']):
                                rslt = df[df['par_num'] == i + 1]
                                try:
                                    sum_lines += rslt.max()['line_num']
                                    # df_orig_par2 = df[df['par_num'] == i]
                                    # df_orig_par2.line_num += df[df['par_num'] == i].max()['line_num']
                                except:
                                    pass
                            line_qt = sum_lines

                        if line_qt == 6 or line_qt == 7:        #Если 6 или 7 линий всего
                            if df.max()['par_num'] == 2:
                                df_temp = df[df.par_num == 2]
                                df_temp.line_num += 3
                                df[df.par_num == 2] = df_temp

                            #   ГЕНЕРИРУЕМ БОКС ДЛЯ РУС ТЕКСТА
                            for i in list(df.index.values):
                                if df['line_num'].iloc[a] == 4 and df['word_num'].iloc[a] == 1:     #Ищем первое слово в 4ой строке
                                    print(df['text'].iloc[a - 1])
                                    y0 = df['top'].iloc[a - 1] + df['height'].iloc[a - 1]           #y0 - координата y под текстом 3 линии
                                    h0 = df['top'].iloc[a] - y0
                                    if h0 < 0:
                                        y0 = df['top'].iloc[a]
                                    else:
                                        y0 = y0 + h0 / 2                                            #y0 - координата чуть выше 4ой линии
                                    print(y0)
                                    break
                                a += 1
                            y1 = df['top'].iloc[-1] + df['height'].iloc[-1] + 10
                            heigth = y1 - y0

                            box = np.array([int(y0), 0, int(y1), int(dict_images_stamp[id].shape[1])])
                            img = create_image_boxed(dict_images_stamp[id], box)


                            df_rus = image_to_df(img, section="text_rus", lang=langrus, config=r'--oem 3 --psm 6 -c page_separator=""')

                            dict_bad_conf_index_to_images = {}

                            df_rus['low'] = df_rus['top'] + df_rus['height']
                            df_rus['right'] = df_rus['left'] + df_rus['width']

                            line1_top_border = df_rus[df_rus['line_num'] == 1].max()['top']         #df вытаскиваем те где лн == 1, выбираем самый большой top
                            line1_low_border = df_rus[df_rus['line_num'] == 1].max()['low']         #df вытаскиваем те где лн == 1, выбираем самый большой top + height (low)
                            line2_top_border = df_rus[df_rus['line_num'] == 2].min()['top']         #df вытаскиваем те где лн == 2, выбираем самый маленький top
                            line2_low_border = df_rus[df_rus['line_num'] == 2].max()['low']         #df вытаскиваем те где лн == 2, выбираем самый большой low
                            line3_top_border = df_rus[df_rus['line_num'] == 3].min()['top']         #df вытаскиваем те где лн == 3, выбираем самый маленький top
                            line3_low_border = df_rus[df_rus['line_num'] == 3].max()['low']         #df вытаскиваем те где лн == 3, выбираем самый маленький low


                            last_line1_num = df_rus[df_rus['line_num'] == 1].max()['word_num']
                            last_line2_num = df_rus[df_rus['line_num'] == 2].max()['word_num']
                            last_line3_num = df_rus[df_rus['line_num'] == 3].max()['word_num']

                            if df_rus.max()['line_num'] == 4:
                                line4_top_border = df_rus[df_rus['line_num'] == 4].min()['top']
                                line4_low_border = df_rus[df_rus['line_num'] == 4].min()['low']

                                last_line4_num = df_rus[df_rus['line_num'] == 4].max()['word_num']

                            #   КОРРЕКЦИЯ РУС ТЕКСТА
                            for word in df_rus['text_rus']:
                                if df_rus.loc[df_rus.text_rus == word, 'conf'].values[0] < 91 or has_numbers(word):
                                    index_row = df_rus.index[df_rus['text_rus'] == word].tolist()[0]
                                    word_num_inline = df_rus.loc[df_rus.text_rus == word, "word_num"][index_row]
                                    line_num = df_rus["line_num"][index_row]
                                    words_in_this_line_qt = len(df_rus[df_rus['line_num'] == line_num])
                                    if re.search(r"^[ЁёА-я/\\\.-]+$+", df_rus['text_rus'][index_row]) is not None:
                                        continue
                                    else:

                                        #   ПОЛУЧАЕМ БОКСЫ
                                        #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В ПЕРВОЙ ЛИНИИ
                                        if df_rus['line_num'][index_row] == 1:
                                            if line1_top_border - 10 < 0:
                                                y0 = 0      #Первая строка
                                            else:
                                                y0 = line1_top_border - 10
                                            if df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[0] == 1:    #Слово первое
                                                w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[df_rus[df_rus['line_num'] == line_num].index[df_rus[df_rus['line_num'] == line_num]['word_num'] == word_num_inline + 1].tolist()[0], 'left']) / 2
                                                if df_rus.loc[df_rus.text_rus == word, "left"] < 10:
                                                    x0 = 0
                                                else:
                                                    x0 = df_rus.loc[df_rus.text_rus == word, "left"] - 10
                                            elif df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[0] == last_line1_num:      #Слово последнее
                                                x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].index[df_rus[df_rus['line_num'] == line_num]['word_num'] == word_num_inline - 1].tolist()[0], 'right'] + df_rus.loc[index_row, 'left']) / 2
                                                if df_rus.loc[df_rus.text_rus == word, "left"].values[0] + df_rus.loc[df_rus.text_rus == word, "width"].values[0] + 10 > img.shape[1]:      #Если близко к краю справа
                                                    w0 = img.shape[1]
                                                else:
                                                    w0 = df_rus.loc[df_rus.text_rus == word, "left"].iloc[0] + df_rus.loc[df_rus.text_rus == word, "width"].iloc[0] + 10
                                            else:   #Посередине
                                                table_with_line = df_rus[df_rus['line_num'] == line_num]
                                                index_of_previous = table_with_line.index[table_with_line['word_num'] == word_num_inline - 1].tolist()[0]
                                                index_of_next = table_with_line.index[table_with_line['word_num'] == word_num_inline + 1].tolist()[0]
                                                x0 = (df_rus.loc[index_of_previous, 'right'] + df_rus.loc[index_row, 'left']) / 2
                                                # x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].word_num == word_num_inline - 1, "left"] +
                                                #       df_rus.loc[df_rus.word_num == word_num_inline - 1, "width"] +
                                                #       df_rus['left'][index_row]) / 2
                                                w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[index_of_next, 'left']) / 2
                                                # w0 = (df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"] + 1, "left"] + df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "width"] - \
                                                #      df_rus.loc[ df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "left"]) / 2

                                            y1 = (line2_top_border - df_rus.loc[df_rus.text_rus == word, "top"].values[0] -       #между y_line1_low  и y_line2_top
                                                  df_rus.loc[df_rus.text_rus == word, "height"].values[0]) / 2 + \
                                                 df_rus.loc[df_rus.text_rus == word, "top"].values[0] + \
                                                 df_rus.loc[df_rus.text_rus == word, "height"].values[0]
                                        #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В 3-ей ЛИНИИ
                                        elif df_rus['line_num'][index_row] == 3:
                                            y0 = line2_low_border
                                            if df_rus.max()['line_num'] == 3:    #Считается последней линией
                                                if word_num_inline == 1 and words_in_this_line_qt > 1:    #Первое слово в строке
                                                    if df_rus["left"][index_row] < 10:
                                                        x0 = 0
                                                    else:
                                                        x0 = df_rus["left"][index_row] - 10

                                                    # index_of_previous_word  = df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]
                                                    w0 = ( df_rus["left"][df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]] + df_rus['right'][index_row] ) / 2       #Берём левую координату следующего слова
                                                elif word_num_inline == last_line3_num and words_in_this_line_qt > 1:     #Последнее
                                                    x0 = ( df_rus["right"][df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row - 1]].tolist()[0]] + df_rus['left'][index_row] ) / 2       #Берём правую координату предыдущего
                                                    if df_rus["right"][index_row] + 10 > img.shape[1]:      #Край справа
                                                        w0 = img.shape[1]
                                                    else:
                                                        w0 = df_rus["right"][index_row] + 10
                                                else:   #Посередине
                                                    if words_in_this_line_qt == 1:                          #Слово единственное (Используем только его коорды)
                                                        if df_rus['left'][index_row] - 10 > 0:
                                                            x0 = df_rus['left'][index_row] - 10
                                                        else:
                                                            x0 = 0
                                                        if df_rus['right'] + 10 < img.shape[1]:
                                                            w0 = df_rus['right'] + 10
                                                        else:
                                                            w0 = img.shape[1]
                                                    else:                                                   #Слово не единственное в строке (Используем соседей)
                                                        df_cut = df_rus[df_rus['line_num'] == line_num]
                                                        x0 = df_cut["right"][df_cut.index[df_cut['word_num'] == df_cut["word_num"][index_row] - 1].tolist()[0]] + 1
                                                        w0 = df_cut["left"][df_cut.index[df_cut['word_num'] == df_cut["word_num"][index_row] + 1].tolist()[0]] - 1

                                                if df_rus["low"][index_row] + 10 >= img.shape[0]:
                                                    y1 = img.shape[0]
                                                else:
                                                    y1 = df_rus["low"][index_row] + 10

                                            elif df_rus.max()['line_num'] == 4:     #Есть 4ая линия
                                                y1 = line4_top_border - 1
                                        elif df_rus['line_num'][index_row] == 4:        #Слово в 4ой линии
                                            pass

                                        try:
                                            box = np.array([int(y0), int(x0), int(y1), int(w0)])
                                            im1 = create_image_boxed(img, box)
                                            text_bad_conf_eng = image_to_df(im1, 'text', lang=langeng, config=r'--oem 3 --psm 7 -c page_separator=""')
                                            text_bad_conf_rus = image_to_df(im1, 'text', lang=langrus, config=r'--oem 3 --psm 7 -c page_separator=""')
                                            dict_bad_conf_index_to_images[index_row] = im1
                                        except:
                                            continue
                                        try:
                                            if has_numbers(str(text_bad_conf_eng['text'].iloc[0])):
                                                df_rus['text_rus'][index_row] = text_bad_conf_eng['text'].iloc[0]
                                            else:
                                                if text_bad_conf_eng['conf'].iloc[0] > text_bad_conf_rus['conf'].iloc[0]:
                                                    if df_rus['conf'][index_row] < text_bad_conf_eng['conf'].iloc[0]:
                                                        df_rus['text'][index_row] = text_bad_conf_eng['text'].iloc[0]
                                                else:
                                                    if df_rus['conf'][index_row] < text_bad_conf_rus['conf'].iloc[0]:
                                                        df_rus['text'][index_row] = text_bad_conf_rus['text'].iloc[0]
                                        except IndexError:
                                            try:
                                                df_cut_2nd_part = df.drop(df[df.line_num <= 3].index)
                                                index_cut = df_cut_2nd_part.index[df_rus['text_rus'] == word].tolist()[0]
                                                if df_rus['conf'][index_row] < df_cut_2nd_part['conf'][index_cut] and df_cut_2nd_part.word_num[index_cut] == df_rus.word_num[index_row]:
                                                    df_rus['text_rus'][index_row] = df['text'][index_row]
                                            except:
                                                pass

                            df = df.drop(df[df.line_num > 3].index)
                            # df[-len(df_rus):] = df_rus
                            text_eng_name = ""
                            text_rus_name = ""
                            for word_space_eng in df[df['line_num'] < 4]['text']:
                                if type(word_space_eng) == float:
                                    word_space_eng = int(word_space_eng)
                                text_eng_name = text_eng_name + str(word_space_eng)
                                text_eng_name = text_eng_name + " "
                            print(text_eng_name)
                            for word_space_rus in df_rus['text_rus']:
                                if type(word_space_rus) == float:
                                    word_space_rus = int(word_space_rus)
                                text_rus_name = text_rus_name + str(word_space_rus)
                                text_rus_name = text_rus_name + " "

                        elif line_qt == 4 or line_qt == 5:          #ЕСЛИ 4 ИЛИ 5 ЛИНИЙ ВСЕГО
                            if df.max()['par_num'] == 2:
                                df_temp = df[df.par_num == 2]
                                df_temp.line_num += 2
                                df[df.par_num == 2] = df_temp

                            for i in list(df.index.values):
                                if df['line_num'].iloc[a] == 3 and df['word_num'].iloc[a] == 1:     #Ищем первое слово в 3ей строке
                                    print(df['text'].iloc[a - 1])
                                    y0 = df['top'].iloc[a - 1] + df['height'].iloc[a - 1]           #y0 - координата y под текстом 2 линии
                                    h0 = df['top'].iloc[a] - y0
                                    if h0 < 0:
                                        y0 = df['top'].iloc[a]
                                    else:
                                        y0 = y0 + h0 / 2                                            #y0 - координата чуть выше 3ей линии
                                    print(y0)
                                    break
                                a += 1
                            y1 = df['top'].iloc[-1] + df['height'].iloc[-1] + 10
                            heigth = y1 - y0

                            box = np.array([int(y0), 0, int(y1), int(dict_images_stamp[id].shape[1])])
                            img = create_image_boxed(dict_images_stamp[id], box)

                            df_rus = image_to_df(img, section="text_rus", lang=langrus,
                                                 config=r'--oem 3 --psm 6 -c page_separator=""')

                            dict_bad_conf_index_to_images = {}

                            df_rus['low'] = df_rus['top'] + df_rus['height']
                            df_rus['right'] = df_rus['left'] + df_rus['width']

                            line1_top_border = df_rus[df_rus['line_num'] == 1].max()[
                                'top']  # df вытаскиваем те где лн == 1, выбираем самый большой top
                            line1_low_border = df_rus[df_rus['line_num'] == 1].max()[
                                'low']  # df вытаскиваем те где лн == 1, выбираем самый большой top + height (low)
                            line2_top_border = df_rus[df_rus['line_num'] == 2].min()[
                                'top']  # df вытаскиваем те где лн == 2, выбираем самый маленький top
                            line2_low_border = df_rus[df_rus['line_num'] == 2].max()[
                                'low']  # df вытаскиваем те где лн == 2, выбираем самый большой low


                            last_line1_num = df_rus[df_rus['line_num'] == 1].max()['word_num']
                            last_line2_num = df_rus[df_rus['line_num'] == 2].max()['word_num']


                            if df_rus.max()['line_num'] == 4:
                                line4_top_border = df_rus[df_rus['line_num'] == 4].min()['top']
                                line4_low_border = df_rus[df_rus['line_num'] == 4].min()['low']
                                line3_top_border = df_rus[df_rus['line_num'] == 3].min()[
                                    'top']  # df вытаскиваем те где лн == 3, выбираем самый маленький top
                                line3_low_border = df_rus[df_rus['line_num'] == 3].max()[
                                    'low']  # df вытаскиваем те где лн == 3, выбираем самый маленький low

                                last_line3_num = df_rus[df_rus['line_num'] == 3].max()['word_num']
                                last_line4_num = df_rus[df_rus['line_num'] == 4].max()['word_num']

                            elif df_rus.max()['line_num'] == 3:
                                line3_top_border = df_rus[df_rus['line_num'] == 3].min()[
                                    'top']  # df вытаскиваем те где лн == 3, выбираем самый маленький top
                                line3_low_border = df_rus[df_rus['line_num'] == 3].max()[
                                    'low']  # df вытаскиваем те где лн == 3, выбираем самый маленький low

                                last_line3_num = df_rus[df_rus['line_num'] == 3].max()['word_num']


                            for word in df_rus['text_rus']:
                                if df_rus.loc[df_rus.text_rus == word, 'conf'].values[0] < 91 or has_numbers(word):
                                    index_row = df_rus.index[df_rus['text_rus'] == word].tolist()[0]
                                    word_num_inline = df_rus.loc[df_rus.text_rus == word, "word_num"][index_row]
                                    line_num = df_rus["line_num"][index_row]
                                    words_in_this_line_qt = len(df_rus[df_rus['line_num'] == line_num])     # Кол-во слов
                                    if re.search(r"^[ЁёА-я/\\\.-]+$+", df_rus['text_rus'][index_row]) is not None:
                                        continue
                                    else:

                                        #   ПОЛУЧАЕМ БОКСЫ
                                        #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В ПЕРВОЙ ЛИНИИ
                                        if df_rus['line_num'][index_row] == 1:
                                            if line1_top_border - 10 < 0:
                                                y0 = 0      #Первая строка
                                            else:
                                                y0 = line1_top_border - 10
                                            if df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[0] == 1:    #Слово первое
                                                w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[df_rus[df_rus['line_num'] == line_num].index[df_rus[df_rus['line_num'] == line_num]['word_num'] == word_num_inline + 1].tolist()[0], 'left']) / 2
                                                if df_rus.loc[df_rus.text_rus == word, "left"] < 10:
                                                    x0 = 0
                                                else:
                                                    x0 = df_rus.loc[df_rus.text_rus == word, "left"] - 10
                                            elif df_rus.loc[df_rus.text_rus == word, "word_num"].iloc[0] == last_line1_num:      #Слово последнее
                                                x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].index[df_rus[df_rus['line_num'] == line_num]['word_num'] == word_num_inline - 1].tolist()[0], 'right'] + df_rus.loc[index_row, 'left']) / 2
                                                if df_rus.loc[df_rus.text_rus == word, "left"].values[0] + df_rus.loc[df_rus.text_rus == word, "width"].values[0] + 10 > img.shape[1]:      #Если близко к краю справа
                                                    w0 = img.shape[1]
                                                else:
                                                    w0 = df_rus.loc[df_rus.text_rus == word, "left"].iloc[0] + df_rus.loc[df_rus.text_rus == word, "width"].iloc[0] + 10
                                            else:   #Посередине
                                                table_with_line = df_rus[df_rus['line_num'] == line_num]
                                                index_of_previous = table_with_line.index[table_with_line['word_num'] == word_num_inline - 1].tolist()[0]
                                                index_of_next = table_with_line.index[table_with_line['word_num'] == word_num_inline + 1].tolist()[0]
                                                x0 = (df_rus.loc[index_of_previous, 'right'] + df_rus.loc[index_row, 'left']) / 2
                                                # x0 = (df_rus.loc[df_rus[df_rus['line_num'] == line_num].word_num == word_num_inline - 1, "left"] +
                                                #       df_rus.loc[df_rus.word_num == word_num_inline - 1, "width"] +
                                                #       df_rus['left'][index_row]) / 2
                                                w0 = (df_rus.loc[index_row, 'right'] + df_rus.loc[index_of_next, 'left']) / 2
                                                # w0 = (df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"] + 1, "left"] + df_rus.loc[df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "width"] - \
                                                #      df_rus.loc[ df_rus.word_num == df_rus.loc[df_rus.text_rus == word, "word_num"], "left"]) / 2

                                            y1 = (line2_top_border - df_rus.loc[df_rus.text_rus == word, "top"].values[0] -       #между y_line1_low  и y_line2_top
                                                  df_rus.loc[df_rus.text_rus == word, "height"].values[0]) / 2 + \
                                                 df_rus.loc[df_rus.text_rus == word, "top"].values[0] + \
                                                 df_rus.loc[df_rus.text_rus == word, "height"].values[0]
                                        #   СЛОВА КОТОРЫЕ НАДО ИСПРАВИТЬ В 3-ей ЛИНИИ
                                        elif df_rus['line_num'][index_row] == 3:
                                            y0 = line2_low_border
                                            if df_rus.max()['line_num'] == 3:    #Считается последней линией
                                                if word_num_inline == 1 and words_in_this_line_qt > 1:    #Первое слово в строке
                                                    if df_rus["left"][index_row] < 10:
                                                        x0 = 0
                                                    else:
                                                        x0 = df_rus["left"][index_row] - 10

                                                    # index_of_previous_word  = df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]
                                                    w0 = ( df_rus["left"][df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row + 1]].tolist()[0]] + df_rus['right'][index_row] ) / 2       #Берём левую координату следующего слова
                                                elif word_num_inline == last_line3_num and words_in_this_line_qt > 1:     #Последнее
                                                    x0 = ( df_rus["right"][df_rus[df_rus['line_num'] == 3].index[df_rus[df_rus['line_num'] == 3]['word_num'] == df_rus['word_num'][index_row - 1]].tolist()[0]] + df_rus['left'][index_row] ) / 2       #Берём правую координату предыдущего
                                                    if df_rus["right"][index_row] + 10 > img.shape[1]:      #Край справа
                                                        w0 = img.shape[1]
                                                    else:
                                                        w0 = df_rus["right"][index_row] + 10
                                                else:   #Посередине
                                                    if words_in_this_line_qt == 1:                          #Слово единственное (Используем только его коорды)
                                                        if df_rus['left'][index_row] - 10 > 0:
                                                            x0 = df_rus['left'][index_row] - 10
                                                        else:
                                                            x0 = 0
                                                        if df_rus['right'] + 10 < img.shape[1]:
                                                            w0 = df_rus['right'] + 10
                                                        else:
                                                            w0 = img.shape[1]
                                                    else:                                                   #Слово не единственное в строке (Используем соседей)
                                                        df_cut = df_rus[df_rus['line_num'] == line_num]
                                                        x0 = df_cut["right"][df_cut.index[df_cut['word_num'] == df_cut["word_num"][index_row] - 1].tolist()[0]] + 1
                                                        w0 = df_cut["left"][df_cut.index[df_cut['word_num'] == df_cut["word_num"][index_row] + 1].tolist()[0]] - 1

                                                if df_rus["low"][index_row] + 10 >= img.shape[0]:
                                                    y1 = img.shape[0]
                                                else:
                                                    y1 = df_rus["low"][index_row] + 10

                                            elif df_rus.max()['line_num'] == 4:     #Есть 4ая линия
                                                y1 = line4_top_border - 1
                                        elif df_rus['line_num'][index_row] == 4:        #Слово в 4ой линии
                                            pass

                                        try:
                                            box = np.array([int(y0), int(x0), int(y1), int(w0)])
                                            im1 = create_image_boxed(img, box)
                                            text_bad_conf_eng = image_to_df(im1, 'text', lang=langeng, config=r'--oem 3 --psm 7 -c page_separator=""')
                                            text_bad_conf_rus = image_to_df(im1, 'text', lang=langrus, config=r'--oem 3 --psm 7 -c page_separator=""')
                                            dict_bad_conf_index_to_images[index_row] = im1
                                        except:
                                            continue
                                        try:
                                            if has_numbers(str(text_bad_conf_eng['text'].iloc[0])):
                                                df_rus['text_rus'][index_row] = text_bad_conf_eng['text'].iloc[0]
                                            else:
                                                if text_bad_conf_eng['conf'].iloc[0] > text_bad_conf_rus['conf'].iloc[0]:
                                                    if df_rus['conf'][index_row] < text_bad_conf_eng['conf'].iloc[0]:
                                                        df_rus['text'][index_row] = text_bad_conf_eng['text'].iloc[0]
                                                else:
                                                    if df_rus['conf'][index_row] < text_bad_conf_rus['conf'].iloc[0]:
                                                        df_rus['text'][index_row] = text_bad_conf_rus['text'].iloc[0]
                                        except IndexError:
                                            try:
                                                df_cut_2nd_part = df.drop(df[df.line_num <= 3].index)
                                                index_cut = df_cut_2nd_part.index[df_rus['text_rus'] == word].tolist()[0]
                                                if df_rus['conf'][index_row] < df_cut_2nd_part['conf'][index_cut] and df_cut_2nd_part.word_num[index_cut] == df_rus.word_num[index_row]:
                                                    df_rus['text_rus'][index_row] = df['text'][index_row]
                                            except:
                                                pass



                        # text_list = text.split("\n")
                        # if len(text_list) >= 6:
                        #     eng_text = text_list[0:3]
                        #     rus_text = text_list[3:len(text_list) - 1]
                        #     print(text_list)
                        #     dict_text["Rus_text"] = "\n".join(rus_text)
                        #     dict_text["Eng_text"] = "\n".join(eng_text)
                        # elif len(text_list) < 6:
                        #     eng_text = text_list[0:2]
                        #     rus_text = text_list[2: len(text_list) - 1]
                        #     print(text_list)
                        #     dict_text["Rus_text"] = "\n".join(rus_text)
                        #     dict_text["Eng_text"] = "\n".join(eng_text)
                        if isinstance(dict_images_stamp[id], list):
                            text_name_pr = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_5+rus1.7to1.9_4",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        else:
                            text_name_pr = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_5+rus1.7to1.9_4",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')

                    elif id in ["Proj_no_h", "Proj_no_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="eng1.3to_proj_no3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.3to_proj_no3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        print("ТЕКСТ ОРИГ ПРОЖ. НОМЕРА:   " + text)
                        patterns0 = [
                            re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                        ]
                        text = replace_code(text)
                        for match_zeros in re.finditer(patterns0[0], text):
                            text = text[0:match_zeros.start()] + text[match_zeros.start(): match_zeros.end()].replace(
                                "0",
                                "O") + text[
                                       match_zeros.end(): len(
                                           text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                        dict_text[id] = text
                    elif id in ["Dr_no_h", "Dr_no_v"]:
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        patterns0 = [
                            re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                        ]
                        print("ТЕКСТ ОРИГ ДР НОМЕРА:    os.path.basename(path)" + text)
                        print("НОМЕР ДОКА : " + os.path.basename(path))
                        text = replace_code(text)
                        for match_zeros in re.finditer(patterns0[0], text):
                            text = text[0:match_zeros.start()] + text[match_zeros.start(): match_zeros.end()].replace(
                                "0",
                                "O") + text[
                                       match_zeros.end(): len(
                                           text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                        dict_text[id] = text
                    else:
                        if isinstance(dict_images_stamp[id], list):
                            list_text = []
                            for i in dict_images_stamp[id]:
                                text = image_to_string(i, lang="eng1.9to1.10_4+rus1.7to1.9_2",
                                                       config=r'--oem 3 --psm 6 -c page_separator=""')
                                list_text.append(text)
                            dict_text[id] = list_text
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_4+rus1.7to1.9_2",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                            dict_text[id] = text
            time_loop0_f = time.time() - time_loop0_b
            print("--- %s seconds for 1 external loop ---" % (time_loop0_f))
            # if id == "Role_input":
            # dict_text[id]["Coordinates"] = #getcoord
        dict_text["File_name"] = os.path.basename(path)
        dict_text["CHECK_WITH_THIS"] = text_name_pr
        dict_text["Rus_name"] = text_rus_name
        dict_text["Eng_name"] = text_eng_name
        time_json_writing_b = time.time()



        ###JSON WRITING
        print(dict_text)
        with open("sample.json", "w", encoding="utf-8") as outfile:
            json.dump(dict_text, outfile, skipkeys=True, indent=4)
        time_json_writing_f1 = time.time() - time_json_writing_b
        print("--- %s seconds for json writing sample.json ---" % (time_json_writing_f1))
        with open("sample_no_ascii.json", "w", encoding="utf-8") as outfile:
            json.dump(dict_text, outfile, skipkeys=True, indent=4, ensure_ascii=False)
        time_json_writing_f2 = time.time() - time_json_writing_f1
        print("--- %s seconds for json writing sample_no_ascii.json ---" % (time_json_writing_f2))
        countstr = ""
        for id in dict_text:
            if id != "File_name":
                if isinstance(dict_text[id], list):
                    for i in dict_text[id]:
                        countstr += str(i.replace("/n", ""))
                else:
                    countstr += str(dict_text[id].replace("/n", ""))
        print(countstr)
        print("length of text = " + str(len(countstr)))
        return r1, dict_images, dict_text
    except:
        question_rep = False
        for id in dict_images:
            if open_window(dict_images[id]):
                question_rep = True
                image_new = dict_images[id]
                break
            else:
                continue
        if question_rep:
            half_tesseract_pdf_new(model1, image_new)

        return r1, dict_images, dict_text


def half_tesseract_pdf_new(model1, image_new):
    what_cropped = ["Stamp", "Company_name", "Label_title_h", "Label_title_v", "Label_Proj_no_h",
                    "Label_Proj_no_v", "Label_dr_no_h", "Label_dr_no_v", "Label_rev_h", "Label_rev_v",
                    "Label_scale_h", "Label_scale_v", "Label_date_h", "Label_date_v"]
    dict_text = {}
    time_loop0_b = time.time()
    r1 = model1.detect([image_new], verbose=0)[0]  # ПОЛУЧАЕМ КОРДЫ И ИМЕНА

    time_model1_detect_f = time.time() - time_loop0_b
    print("--- %s seconds for second detection stamp ---" % (time_model1_detect_f))
    image_pure, rfederaciya = crop_instances(image_new, r1['rois'], r1['class_ids'],
                                             [1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 23, 24])

    # rfederaciya = what_to_crop
    im = Image.fromarray(image_pure)
    im.show()
    dict_images_stamp = instances_to_images(image_pure, r1['rois'], r1['class_ids'], names=class_names_model1,
                                            list_accepted_ids=[5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 27, 28,
                                                               29, 30, 31, 32, 33, 34], stamp=True)
    for id in dict_images_stamp:
        if id in what_cropped:
            continue
        else:
            if id == "Date_v" or id == "Date_h":
                if isinstance(dict_images_stamp[id], list):
                    text = datetime_extract(dict_images_stamp[id][0])
                else:
                    text = datetime_extract(dict_images_stamp[id])
                dict_text[id] = text
            elif id in ["Label_oper", "Label_eng", "Label_mgr", "Label_chk", "Label_by", "Label_supv"]:
                try:
                    if dict_images_stamp[id]["Coordinates"] is None:
                        dict_text[id] = "Empty"
                    else:
                        text = image_to_string(dict_images_stamp[id]["Coordinates"], lang="eng1.9to1.10_3",
                                               config=r'--oem 3 --psm 7 -c page_separator=""')
                        print("ОРИГ ЛЭЙБЛА РОЛИ: " + text)
                        text = check_O(text)
                        text = replace_numbers(text)
                        dict_text[id] = text

                except TypeError:
                    if isinstance(dict_images_stamp[id], list):
                        if dict_images_stamp[id][0]["Coordinates"] is None:
                            dict_text[id] = "Empty"
                        else:
                            im = Image.fromarray(dict_images_stamp[id][0]["Coordinates"])
                            dict_text[id] = replace_numbers(check_O(image_to_string(im, lang="eng1.9to1.10_3",
                                                                                    config=r'--oem 3 --psm 7 -c '
                                                                                           r'page_separator=""')))
            elif id == "Role_input":
                pass
            elif id in ["REV_h", "REV_v"]:
                if isinstance(dict_images_stamp[id], list):
                    text = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_3",
                                           config=r'--oem 3 --psm 7 -c page_separator=""')
                else:
                    text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_3",
                                           config=r'--oem 3 --psm 7 -c page_separator=""')
                text = text.replace("O", "0")
                dict_text[id] = text
            elif id in ["Scale_v", "Scale_h"]:
                if isinstance(dict_images_stamp[id], list):
                    text = image_to_string(dict_images_stamp[id][0], lang="eng1.9+rus1.7to1.9_2",
                                           config=r'--oem 3 --psm 7 -c page_separator=""')
                else:
                    text = image_to_string(dict_images_stamp[id], lang="eng1.9+rus1.7to1.9_2",
                                           config=r'--oem 3 --psm 7 -c page_separator=""')
                if text == "":
                    text = "-"
                dict_text[id] = text
            elif id in ["Project_name_h", "Project_name_v"]:
                if isinstance(dict_images_stamp[id], list):
                    text = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_3+rus1.7to1.9_2",
                                           config=r'--oem 3 --psm 6 -c page_separator=""')
                else:
                    text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_3+rus1.7to1.9_2",
                                           config=r'--oem 3 --psm 6 -c page_separator=""')
                patterns0 = [
                    re.compile(r"\b0-|-0-")  # Паттерн отдельных мнимых 0, в реальности это буква О
                ]
                patterns = [
                    re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                    # Паттерны поиска Dr no или Proj no внутри Project_name
                ]
                listx = [x.group() for x in re.finditer(patterns[0], text)]  # Лист из всех найденных кодов
                for x in listx:  # Для каждого кода в Project_name
                    if "0" not in x:
                        continue
                    x1 = x
                    # По каждому паттерну: 0- или -0-
                    for match_zeros in re.finditer(patterns0[0], x):
                        x1 = x[0:match_zeros.start()] + x[match_zeros.start(): match_zeros.end()].replace("0",
                                                                                                          "O") + x[
                                                                                                                 match_zeros.end(): len(
                                                                                                                     x)]
                    try:
                        text = text.replace(x, x1)
                    except UnboundLocalError:
                        pass
                dict_text[id] = text
            elif id in ["Proj_no_h", "Proj_no_v"]:
                if isinstance(dict_images_stamp[id], list):
                    text = image_to_string(dict_images_stamp[id][0], lang="Proj_no-1",
                                           config=r'--oem 3 --psm 7 -c page_separator=""')
                else:
                    text = image_to_string(dict_images_stamp[id], lang="Proj_no_1",
                                           config=r'--oem 3 --psm 7 -c page_separator=""')
                print("ТЕКСТ ОРИГ ПРОЖ. НОМЕРА:   " + text)
                patterns0 = [
                    re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                    re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                ]
                text = replace_code(text)
                for match_zeros in re.finditer(patterns0[0], text):
                    text = text[0:match_zeros.start()] + text[match_zeros.start(): match_zeros.end()].replace("0",
                                                                                                              "O") + text[
                                                                                                                     match_zeros.end(): len(
                                                                                                                         text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                dict_text[id] = text
            elif id in ["Dr_no_h", "Dr_no_v"]:
                if isinstance(dict_images_stamp[id], list):
                    Image.fromarray(dict_images_stamp[id][0]).show()
                    text = image_to_string(dict_images_stamp[id][0], lang="Dr_no_2",
                                           config=r'--oem 3 --psm 6 -c page_separator=""')
                else:
                    Image.fromarray(dict_images_stamp[id]).show()
                    text = image_to_string(dict_images_stamp[id], lang="Dr_no_2",
                                           config=r'--oem 3 --psm 6 -c page_separator=""')
                patterns0 = [
                    re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                    re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                ]
                text = replace_code(text)
                for match_zeros in re.finditer(patterns0[0], text):
                    text = text[0:match_zeros.start()] + text[match_zeros.start(): match_zeros.end()].replace("0",
                                                                                                              "O") + text[
                                                                                                                     match_zeros.end(): len(
                                                                                                                         text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                dict_text[id] = text
            else:
                if isinstance(dict_images_stamp[id], list):
                    list_text = []
                    for i in dict_images_stamp[id]:
                        Image.fromarray(i).show()
                        text = image_to_string(i, lang="eng1.9to1.10_3+rus1.7to1.9_2",
                                               config=r'--oem 3 --psm 6 -c page_separator=""')
                        list_text.append(text)
                    dict_text[id] = list_text
                else:
                    text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_3+rus1.7to1.9_2",
                                           config=r'--oem 3 --psm 6 -c page_separator=""')
                    dict_text[id] = text
            if id != "Role_input":
                if isinstance(dict_text[id], str):
                    print(dict_text[id])
                else:
                    print("НЕ БРАТОК НЕ ПОЛУЧИТСЯ")
    with open("sample_no_ascii.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4, ensure_ascii=False)


def open_window(image_bytes):
    Image.fromarray(image_bytes).save("sample_image.PNG", "PNG")
    layout = [[sg.Text("Это штамп?", key="new")],
              [sg.Image(source=os.getcwd() + "\\sample_image.PNG")],
              [sg.Button('Да', key='-YES-'), sg.Button('Нет', key='-NO-')]
              ]
    window = sg.Window("Yes or No?", layout, modal=True)
    choice = None
    answer = False
    while True:
        event, values = window.read()
        if event == '-YES-':
            answer = True
            break
        if event == '-NO-':
            answer = False
            break
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
    if answer:
        return True
    else:
        return False


def get_first_key(dictionary):
    for key in dictionary:
        return key
    raise IndexError


def do_multiple(folder, model1, model2, language_comb):
    WHAT_CROPPED = ["Stamp", "Company_name", "Label_title_h", "Label_title_v", "Label_Proj_no_h",
                    "Label_Proj_no_v", "Label_dr_no_h", "Label_dr_no_v", "Label_rev_h", "Label_rev_v",
                    "Label_scale_h", "Label_scale_v", "Label_date_h", "Label_date_v"]

    time_beginning = time.time()

    list_dicts = []
    for pdf_base_n in os.listdir(folder):
        if pdf_base_n.endswith(".pdf"):
            print("============================================НОВЫЙ ДОК============================================")
            time_image_pre = time.time()
            image = get_image_for_blueprint(folder + "\\" + pdf_base_n, dpi=300)
            time_image_pre_fin = time.time()
            print("Time consumed by image_pre : %s" % (float(time_image_pre_fin - time_image_pre)))

            time_1st_mask = time.time()
            r = model2.detect([image], verbose=0)[0]

            # Корректируем детект чертежа
            if 1 in r['class_ids'] and 2 in r['class_ids']:  # Оба штампа найдены
                # Выбираем тот в котором больше уверенности
                if r['scores'][np.where(r['class_ids'] == 1)[0][0]] > r['scores'][np.where(r['class_ids'] == 2)[0][0]]:
                    accept = [1]
                else:
                    accept = [2]
            else:
                accept = [1, 2]

            dict_images = instances_to_images(image, r['rois'], r['class_ids'], names=class_names_model2,
                                              list_accepted_ids=accept, stamp=False)
            time_1st_mask_fin = time.time()
            print("Time consumed detecting Blueprint : %s" % (float(time_1st_mask_fin - time_1st_mask)))

            time_2nd_mask = time.time()
            r1 = model1.detect([dict_images[get_first_key(dict_images)]], verbose=0)[0]  # ПОЛУЧАЕМ КОРДЫ И ИМЕНА

            list_dups = array_indexes_of_duplicates_of(r1['class_ids'])  # r['class_ids'] преобразуем в лист
            # Корректируем детект штампа
            r1['scores'], r1['rois'], r1['class_ids'], r1['masks'] = filter_array_of_array(r1['class_ids'],
                                                                                           r1['scores'], r1['rois'],
                                                                                           r1['masks'], list_dups)

            image_pure, rfederaciya = crop_instances(dict_images[get_first_key(dict_images)], r1['rois'],
                                                     r1['class_ids'],
                                                     [1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 23, 24])
            dict_images_stamp = instances_to_images(image_pure, r1['rois'], r1['class_ids'],
                                                    names=class_names_model1,
                                                    list_accepted_ids=[5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26,
                                                                       27, 28,
                                                                       29, 30, 31, 32, 33, 34], stamp=True)
            time_2nd_mask_fin = time.time()
            print("Time consumed by STAMP recognition : %s" % (float(time_2nd_mask_fin - time_2nd_mask)))
            dict_text = {}
            for id in dict_images_stamp:
                if id in WHAT_CROPPED:
                    continue
                else:
                    if id == "Date_v" or id == "Date_h":
                        time_date_text = time.time()
                        if isinstance(dict_images_stamp[id], list):
                            text = datetime_extract(dict_images_stamp[id][0])
                        else:
                            text = datetime_extract(dict_images_stamp[id])
                        # dict_text[id] = text
                        time_date_text_fin = time.time()
                        print("Time consumed by date extraction : %s" %(int(time_date_text_fin - time_date_text)))

                    elif id in ["Label_oper", "Label_eng", "Label_mgr", "Label_chk", "Label_by", "Label_supv"]:
                        time_roles_text = time.time()
                        try:
                            if dict_images_stamp[id]["Coordinates"] is None:
                                dict_text[id] = "Empty"
                            else:
                                text = image_to_string(dict_images_stamp[id]["Coordinates"], lang="eng1.9to1.10_3",
                                                       config=r'--oem 3 --psm 7 -c page_separator=""')
                                # print("ОРИГ ЛЭЙБЛА РОЛИ: " + text)
                                text = check_O(text)
                                text = replace_numbers(text)
                                # dict_text[id] = text
                        except TypeError:
                            if isinstance(dict_images_stamp[id], list):
                                if dict_images_stamp[id][0]["Coordinates"] is None:
                                    # dict_text[id] = "Empty"
                                    pass
                                else:
                                    im = Image.fromarray(dict_images_stamp[id][0]["Coordinates"])
                                    # dict_text[id] = replace_numbers(check_O(image_to_string(im, lang="eng1.9to1.10_3",
                                    #                                                         config=r'--oem 3 --psm 7 -c '
                                    #                                                                r'page_separator=""')))
                        time_roles_text_fin = time.time()
                        print("Time_consumed by 1 role extraction : %s" % (int(time_roles_text_fin - time_roles_text)))

                    elif id == "Role_input":
                        continue

                    elif id in ["REV_h", "REV_v"]:
                        time_rev_text = time.time()
                        if isinstance(dict_images_stamp[id], list):
                            # text = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_3",
                            #                        config=r'--oem 3 --psm 7 -c page_separator=""')
                            # text = image_to_string(dict_images_stamp[id][0], lang="eng1.4to_proj_no3",
                            #                        config=r'--oem 3 --psm 7 -c page_separator=""')
                            text = image_to_string(dict_images_stamp[id][0], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        text = text.replace("O", "0")
                        # dict_text[id] = text
                        time_rev_text_fin = time.time()
                        print("Time consumed by rev extraction : %s" % (int(time_rev_text_fin - time_rev_text)))

                    elif id in ["Scale_v", "Scale_h"]:
                        time_scale_text = time.time()
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="eng1.9+rus1.7to1.9",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.9+rus1.7to1.9",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        if text == "":
                            text = "-"
                        elif text is None:
                            text = "-"
                        # dict_text["Scale"] = text
                        time_scale_text_fin = time.time()
                        print("Time consumed by SCALE extraction : %s" % (int(time_scale_text_fin - time_scale_text)))

                    elif id in ["Project_name_h", "Project_name_v"]:
                        time_project_name_text = time.time()
                        Image.fromarray(dict_images_stamp[id]).show()
                        if isinstance(dict_images_stamp[id], list):
                            # text = image_to_string(dict_images_stamp[id][0], lang="eng1.9to1.10_5+rus1.7to1.9_3",
                            #                        config=r'--oem 3 --psm 6 -c page_separator=""')
                            text = image_to_string(dict_images_stamp[id][0], lang=language_comb,
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang=language_comb,
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                        patterns0 = [
                            re.compile(r"\b0-|-0-")  # Паттерн отдельных мнимых 0, в реальности это буква О
                        ]
                        patterns = [
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                            # Паттерны поиска Dr no или Proj no внутри Project_name
                        ]
                        listx = [x.group() for x in re.finditer(patterns[0], text)]  # Лист из всех найденных кодов
                        for x in listx:  # Для каждого кода в Project_name
                            if "0" not in x:
                                continue
                            x1 = x
                            # По каждому паттерну: 0- или -0-
                            for match_zeros in re.finditer(patterns0[0], x):
                                x1 = x[0:match_zeros.start()] + x[match_zeros.start(): match_zeros.end()].replace("0",
                                                                                                                  "O") + x[
                                                                                                                         match_zeros.end(): len(
                                                                                                                             x)]
                            try:
                                text = text.replace(x, x1)
                            except UnboundLocalError:
                                pass
                        dict_text[id] = text
                        time_project_name_text_fin = time.time()
                        print("ТЕКСТ ФУЛ : %s" %text)
                        print("Time consumed by PROJECT NAME extraction : %s" %(
                            int(time_project_name_text_fin - time_project_name_text)))

                    elif id in ["Proj_no_h", "Proj_no_v"]:
                        time_proj_no_text = time.time()
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                            # text = image_to_string(dict_images_stamp[id], lang="Pr_no_4_from_1.4",
                            #                        config=r'--oem 3 --psm 7 -c page_separator=""')
                        # print("ТЕКСТ ОРИГ ПРОЖ. НОМЕРА:   " + text)
                        patterns0 = [
                            re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                        ]
                        text = replace_code(text)
                        for match_zeros in re.finditer(patterns0[0], text):
                            text = text[0:match_zeros.start()] + text[match_zeros.start(): match_zeros.end()].replace(
                                "0",
                                "O") + text[
                                       match_zeros.end(): len(
                                           text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                        # dict_text[id] = text
                        time_proj_no_text_fin = time.time()
                        print("Time consumed by PROJ NO extraction : %s" %(
                            float(time_proj_no_text_fin - time_proj_no_text)))

                    elif id in ["Dr_no_h", "Dr_no_v"]:
                        time_dr_no_text = time.time()
                        if isinstance(dict_images_stamp[id], list):
                            text = image_to_string(dict_images_stamp[id][0], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="Dr_no_3",
                                                   config=r'--oem 3 --psm 7 -c page_separator=""')
                        patterns0 = [
                            re.compile(r"\b0-|-0-"),  # Паттерн отдельных мнимых 0, в реальности это буква О
                            re.compile(r"([A-Za-z0-9]+(-[A-Za-z0-9]+)+)")
                        ]
                        # print("ТЕКСТ ОРИГ ДР НОМЕРА:    " + text)
                        print("НАЗВАНИЕ ДОКА : " + os.path.basename(folder))
                        text = replace_code(text)
                        for match_zeros in re.finditer(patterns0[0], text):
                            text = text[0:match_zeros.start()] + text[match_zeros.start(): match_zeros.end()].replace(
                                "0",
                                "O") + text[
                                       match_zeros.end(): len(
                                           text)]  # На каждый найденный мнимый 0, заменяем его в тексте
                        # dict_text[id] = text
                        time_dr_no_text_fin = time.time()
                        print("Time consumed by DR NO extraction : %s" % (int(time_dr_no_text_fin - time_dr_no_text)))

                    else:
                        if isinstance(dict_images_stamp[id], list):
                            list_text = []
                            for i in dict_images_stamp[id]:
                                Image.fromarray(i).show()
                                text = image_to_string(i, lang="eng1.9to1.10_4+rus1.7to1.9_2",
                                                       config=r'--oem 3 --psm 6 -c page_separator=""')
                                list_text.append(text)
                            dict_text[id] = list_text
                        else:
                            text = image_to_string(dict_images_stamp[id], lang="eng1.9to1.10_4+rus1.7to1.9_2",
                                                   config=r'--oem 3 --psm 6 -c page_separator=""')
                            dict_text[id] = text

            dict_text["File_name"] = pdf_base_n

            if "Scale_h" not in dict_text and "Scale_v" not in dict_text:
                dict_text["Scale"] = "-"

            list_dicts.append(dict_text)

            # ###SPLASH
            # time_splash = time.time()
            # if dict_images.get('STAMP_h') is not None:
            #     image = image_mask_and_boxes(dict_images['STAMP_h'], r1['rois'], r1['masks'], r1['class_ids'],
            #                                  class_names_model1, r1['scores'])
            # elif dict_images.get('STAMP_v') is not None:
            #     image = image_mask_and_boxes(dict_images['STAMP_v'], r1['rois'], r1['masks'], r1['class_ids'],
            #                                  class_names_model1, r1['scores'])
            # im = Image.fromarray(image)
            # im.show()
            # time_splash_fin = time.time()
            # print("Time_consumed by SPLASH apply : " + str(int(time_splash_fin - time_splash)))

            # if dict_images.get('STAMP_h') is not None:
            #     Image.fromarray(dict_images['STAMP_h']).show()
            # elif dict_images.get('STAMP_v') is not None:
            #     Image.fromarray(dict_images['STAMP_v']).show()

    ###JSON WRITING
    with open("sample_all.json", "a", encoding="utf-8") as outfile:
        for i in list_dicts:
            json.dump(i, outfile, skipkeys=True, indent=4, ensure_ascii=False)


def generate_txt_for_1_line_data(folder_path):
    for i in os.listdir(folder_path):
        if i.endswith(".png"):
            with open(folder_path + "\\" + i.replace(".png", ".gt.txt"), encoding="utf-8") as file:
                pass
                file.write(image_to_string(i, lang="eng1.6+rus1.4", config=r'--oem 3 --psm 6 -c page_separator=""'))


def full_tesseract_pdf_other(model2, path):
    image = get_image_for_blueprint(path)

    r0 = model2.detect([image], verbose=0)[0]
    print(r0['class_ids'])
    n_instances = r0['rois'].shape[0]
    dict_boxed_images = {}
    for i in range(n_instances):
        if not np.any(r0['rois'][i]) or not r0['class_ids'][i] in [4, 3, 5, 6, 7, 9]:
            continue
        box = r0['rois'][i]
        boxed_image = create_image_boxed(image, box)

        dict_boxed_images[r0['class_ids'][i]] = boxed_image

    dict_text = {}
    for key in dict_boxed_images:
        if key == 4:
            # rot_img = dict_boxed_images[key].swapaxes(-2, -1)[..., ::-1]
            # rot_img = np.rot90(dict_boxed_images[key], axes=(-2, -1))
            rot_img = scipy.ndimage.rotate(dict_boxed_images[key], -90)
            print(rot_img)
            # plt.imshow(rot_img, interpolation='nearest')
            # plt.show()
            text = image_to_string(rot_img, lang="eng1.9+rus1.8", config=r'--oem 3 --psm 6 -c page_separator=""')
            dict_text[int(key)] = text
        else:
            # plt.imshow(dict_boxed_images[key], interpolation='nearest')
            # plt.show()
            text = image_to_string(dict_boxed_images[key], lang="eng1.3+rus1.3",
                                   config=r'--oem 3 --psm 6 -c page_separator=""')
            dict_text[int(key)] = text

    print(dict_text)
    with open("sample_other.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4)
    with open("sample_no_ascii_other.json", "w", encoding="utf-8") as outfile:
        json.dump(dict_text, outfile, skipkeys=True, indent=4, ensure_ascii=False)


layout = [
    [sg.Text("Input weights Mask1"), sg.Input(key="-W1-"), sg.FileBrowse(file_types=(("h5 files", "*.h5*"),))],
    [sg.Text("Input weights Mask2"), sg.Input(key="-W2-"), sg.FileBrowse(file_types=(("h5 files", "*.h5*"),))],
    [sg.Text("Input image"), sg.Input(key="-IM-"), sg.FileBrowse()],
    [sg.Text("Input pdf"), sg.Input(key="-PDF-", size=(250, 2)), sg.FileBrowse(file_types=(("pdf docs", "*.pdf*"),))],
    [sg.Text("Input folder"), sg.Input(key="-FL-"),
     sg.FolderBrowse(button_text="Choose the images folder for --data_generation--")],
    [sg.Text("Input language ENG"), sg.Input(key="-L1-"), sg.FileBrowse(file_types=(("traineddata", "*.traineddata*"),)),
     sg.Text("Input language RUS"), sg.Input(key="-L2-"), sg.FileBrowse(file_types=(("traineddata", "*.traineddata*"),))],
    [sg.Button("Load weights1", key="B1", disabled_button_color="blue", button_color="red"),
     sg.Button("Load weights2", key="B2", disabled_button_color="blue", button_color="red")],
    [sg.Button("Generate data for Mask1 from Mask2", key="B6", disabled_button_color="blue", button_color="black"),
     sg.Button("Generate data for tesseract from pdf", key="B12", disabled_button_color="blue", button_color="black")],
    [sg.Button("Generate data for tesseract(model1 required)", key="B8", disabled_button_color="blue",
               button_color="black"),
     sg.Button("Generate data for model2", key="B9", disabled_button_color="blue", button_color="black")],
    [sg.Button("Tesseract from pdf", key="B10", disabled_button_color="blue", button_color="green"),
     sg.Button("Tesseract from pdf (Other)", key="B11", disabled_button_color="blue", button_color="green")],
    [sg.Button("Test call on multiple docs", key="B13", button_color="gold", border_width=15), sg.Button("Show splash(only after \"Tesseract from pdf\")", key="B10_2", button_color="green", border_width=15)],
    [sg.Exit()]
]

window = sg.Window("Models tester", layout, background_color="green")

while True:
    event, values = window.read()
    print(event, values)
    if event in (sg.WINDOW_CLOSED, "Exit"):
        break
    if event == "B1":
        if is_valid_path_weights(values["-W1-"]):
            try:
                model1
                sg.popup_error("Already loaded weights!")
            except NameError:
                time_b1 = time.time()
                config1 = InferenceConfigStamp2()
                config1.display()
                model1 = modellib.MaskRCNN(mode="inference", config=config1,
                                           model_dir=DEFAULT_LOGS_DIR)
                model1.load_weights(values["-W1-"], by_name=True)
                class_names_model1 = ["FONT", "Stamp", "Company_name", "Label_title_h", "Label_title_v",
                                      "Project_name_h", "Project_name_v", "Label_Proj_no_h",
                                      "Label_Proj_no_v", "Proj_no_h", "Proj_no_v", "Label_dr_no_h", "Label_dr_no_v",
                                      "Dr_no_h",
                                      "Dr_no_v", "Label_rev_h", "Label_rev_v", "REV_h", "REV_v", "Label_scale_h",
                                      "Label_scale_v", "Scale_h", "Scale_v", "Label_date_h", "Label_date_v", "Date_h",
                                      "Date_v", "Label_by", "Label_chk", "Label_eng",
                                      "Label_supv", "Label_oper", "Label_mgr", "Role_input", "Blue_drawing_no"]
                # name_dict = {"Stamp": 1, "Company_name": 2, "Label_title_h": 3, "Label_title_v": 4, "Project_name_h": 5,
                #              "Project_name_v": 6, "Label_Proj_no_h": 7,
                #              "Label_Proj_no_v": 8, "Proj_no_h": 9, "Proj_no_v": 10, "Label_dr_no_h": 11,
                #              "Label_dr_no_v": 12, "Dr_no_h": 13,
                #              "Dr_no_v": 14, "Label_rev_h": 15, "Label_rev_v": 16, "REV_h": 17, "REV_v": 18,
                #              "Label_scale_h": 19,
                #              "Label_scale_v": 20, "Scale_h": 21, "Scale_v": 22, "Label_date_h": 23, "Label_date_v": 24,
                #              "Date_h": 25, "Date_v": 26, "Label_by": 27,
                #              "Label_chk": 28, "Label_eng": 29, "Label_supv": 30, "Label_oper": 31, "Label_mgr": 32,
                #              "Role_input": 33, "Blue_drawing_no": 34}
                time_ex_b1 = time.time() - time_b1
                sg.popup_notify("--- %s seconds for 1st model load ---" % (time_ex_b1))

    if event == "B2":
        if is_valid_path_weights(values["-W2-"]):
            try:
                model2
                sg.popup_error("Already loaded weights!")
            except NameError:
                time_b2 = time.time()
                config2 = InferenceConfigBlueprint2()
                config2.display()
                model2 = modellib.MaskRCNN(mode="inference", config=config2,
                                           model_dir=DEFAULT_LOGS_DIR)
                model2.load_weights(values["-W2-"], by_name=True)
                # class_names_model2 = class_names2 = ['FONT', 'STAMP_RECT', 'STAMP_SQUARE', 'NOTES', 'LEGEND',
                #                                      'REFERENCE DRAWINGS', 'DRAWING NO', 'KEY PLAN', 'COMPAS',
                #                                      'PAPER SIZE',
                #                                      'SCALE BAR', 'DRAWING LIMIT N', 'DRAWING LIMIT E', 'HOLDS',
                #                                      'LOCATION', 'TAG',
                #                                      'NEW STAMP DRAWING NO', 'VOID', 'UNSORTED']
                # class_names_model2 = class_names2 = ['FONT', 'STAMP_h', 'STAMP_v', 'COMPAS', 'DRAWING NO',
                #                                      'PAPER_SIZE', 'AD_INFO_h', 'AD_INFO_V', 'Drawing_no_blue', 'SCALE_BAR']
                class_names_model2 = ['FONT', 'STAMP_h', 'STAMP_v', 'COMPAS', 'DRAWING NO',
                                      'PAPER_SIZE', 'Drawing_no_blue', 'SCALE_BAR']  # Blueprint 2.0 no v_h
                time_ex_b2 = time.time() - time_b2
                sg.popup_notify("--- %s seconds for 2nd model load ---" % (time_ex_b2))

    if event == "B6":
        try:
            model2
            if is_valid_path_image(values["-FL-"]):
                generate_data_stamp(model2, values["-FL-"])
        except NameError:
            sg.popup_error("Load weights №2 first")

    if event == "B7":
        try:
            model1
            try:
                t0 = time.time()
                model2
                if is_valid_path_image(values["-IM-"]):
                    full_tesseract(model1, model2, values["-IM-"])
                time_ex_b7 = time.time() - t0
                sg.popup_notify("--- %s seconds for full conversion ---" % (time_ex_b7))
            except NameError:
                sg.popup_error("Load weights №2 first")
        except NameError:
            sg.popup_error("Load weights №1 first")

    if event == "B8":
        try:
            model1
            if is_valid_path_image(values["-FL-"]):
                generate_data_tesseract(model1, values["-FL-"])
        except NameError:
            sg.popup_error("Load weights №1 first")

    if event == "B9":
        if is_valid_path_image(values["-FL-"]):
            generate_data_for_Mask2(values["-FL-"])

    if event == "B10":
        time0 = time.time()
        print(values["-PDF-"])
        try:
            lang_temp = os.path.basename(values["-L1-"]).replace(".traineddata", "") + "+" + os.path.basename(
                values["-L2-"]).replace(".traineddata", "")
            print(lang_temp)
            res, dict_img, _ = full_tesseract_pdf_new(model1, model2, values["-PDF-"], lang_temp, os.path.basename(values['-L1-']).replace(".traineddata", ""), os.path.basename(values['-L2-']).replace(".traineddata", ""))

        except:
            sg.popup_notify("ТЫ ЧТО-ТО НЕ ТАК НАМУТИЛ")
        time_ex_b10 = time.time() - time0
        sg.popup_notify("--- %s seconds for full conversion ---" % (time_ex_b10))

    if event == "B10_2":
        try:
            ###SPLASH
            if dict_img.get('STAMP_h') is not None:
                image = image_mask_and_boxes(dict_img['STAMP_h'], res['rois'], res['masks'], res['class_ids'],
                                             class_names_model1, res['scores'])
            elif dict_img.get('STAMP_v') is not None:
                image = image_mask_and_boxes(dict_img['STAMP_v'], res['rois'], res['masks'], res['class_ids'],
                                             class_names_model1, res['scores'])
            im = Image.fromarray(image)
            im.show()
        except:
            sg.popup_error("Не получается")

    if event == "B11":
        time0 = time.time()
        print(values["-PDF-"])
        full_tesseract_pdf_other(model2, values["-PDF-"])
        time_ex_b10 = time.time() - time0
        sg.popup_notify("--- %s seconds for full conversion ---" % (time_ex_b10))

    if event == "B12":
        if is_valid_path_image(values["-FL-"]):
            generate_data_tesseract_from_pdf_foler(model1, model2, values["-FL-"])

    if event == "B13":
        try:
            os.listdir(values["-FL-"])
            lang_temp = os.path.basename(values["-L1-"]).replace(".traineddata", "") + "+" + os.path.basename(values["-L2-"]).replace(".traineddata", "")
            print(lang_temp)
            do_multiple(values["-FL-"], model1, model2, lang_temp)
        except:
            sg.popup_notify("NE TUDA TIKNUL")

window.close()
