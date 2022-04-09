import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from IPython.display import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model-type", dest="mdlType", required=True)
parser.add_argument("--classes", dest="classes", required=True, type=str)
parser.add_argument("--color-codes", dest="colors", default=None, type=str)
parser.add_argument("--model-save-path", dest="mdlSavePath", required=True)
parser.add_argument("--img-height", dest="imgHeight", default=480, type=int)
parser.add_argument("--img-width", dest="imgWidth", default=640, type=int)
parser.add_argument("--epochs", dest="epochs", default=2, type=int)
parser.add_argument("--steps-per-epoch", dest="stepsPerEpochs", default=512, type=int)
parser.add_argument("--val-steps-per-epoch", dest="valStepsPerEpoch", default=64, type=int)

parser.add_argument("--test-images-path", dest="testImagesPath", required=True, type=str)
parser.add_argument("--results-save-path", dest="resultsSavePath", default=None, type=str)

args = parser.parse_args()

# print(args)
classes = args.classes.split(",")
color_codes = [tuple([int(j) for j in i.split(",")]) for i in args.colors.split(" ")]
test_path = args.testImagesPath + "\\"
res_save_path = args.resultsSavePath
class_num = len(classes)
multi_image_flag = True
if 2 < len(test_path) - test_path.rfind(".") < 5:
    multi_image_flag = False
if res_save_path is None:
    res_save_path = args.mdlSavePath + '\\results\\'

model_name = args.mdlType
if model_name == "vgg_unet":
    from keras_segmentation.models.unet import vgg_unet

    model = vgg_unet(n_classes=class_num, input_height=args.imgHeight, input_width=args.imgWidth)
elif model_name == "segnet":
    from keras_segmentation.models.segnet import segnet

    model = segnet(n_classes=class_num, input_height=args.imgHeight, input_width=args.imgWidth)
# elif model_name == "densenet121_unet_dt" or model_name == "densenet121_unet_dt_big_dataset":
#     from keras_segmentation.models.unet import densenet_unet
#     model = densenet_unet(n_classes=2, input_height=480, input_width=640)
else:
    raise Exception("UnknownModelError")


if not os.path.exists(res_save_path):
    print("save directory doesn't exists, creating...")
    os.mkdir(res_save_path)
if not multi_image_flag:
    print("prediction for single image")
    res_img_name = test_path[test_path.rfind("\\") + 1:test_path.rfind(".")]
    results_name = res_save_path + "out_" + res_img_name

    out = model.predict_segmentation(
        inp=test_path,
        out_fname=results_name + ".png",
        colors=color_codes
    )

    # Image(results_name + ".png")

    out_segment = model.predict_segmentation(
        inp=test_path,
        out_fname=results_name + "_segment.png", overlay_img=True, show_legends=True,
        class_names=classes,
        colors=color_codes

    )

    # Image(results_name + "_segment.png")

    print("resulting images successfully saved")
else:
    print("predicting for multiple image")
    print("working in '" + test_path + "' directory")
    from glob import glob
    for name in glob(test_path + "*.png"):
        res_img_name = name[name.rfind("\\") + 1:name.rfind(".")]
        results_name = res_save_path + "out_" + res_img_name

        out = model.predict_segmentation(
            inp=name,
            out_fname=results_name + ".png",
            colors=color_codes
        )

        out_segment = model.predict_segmentation(
            inp=name,
            out_fname=results_name + "_segment.png", overlay_img=True, show_legends=True,
            class_names=classes,
            colors=color_codes

        )

        print("saving results for '" + res_img_name + "'")