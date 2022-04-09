import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model-type", dest="mdlType", required=True)
parser.add_argument("--class-num", dest="classNum", required=True, type=int)
parser.add_argument("--model-save-path", dest="mdlSavePath", required=True)
parser.add_argument("--img-height", dest="imgHeight", default=480, type=int)
parser.add_argument("--img-width", dest="imgWidth", default=640, type=int)
parser.add_argument("--epochs", dest="epochs", default=2, type=int)
parser.add_argument("--steps-per-epoch", dest="stepsPerEpochs", default=512, type=int)
parser.add_argument("--val-steps-per-epoch", dest="valStepsPerEpoch", default=64, type=int)

parser.add_argument("--train-images-path", dest="trainImagesPath", required=True)
parser.add_argument("--train-annotations-path", dest="trainAnnotationsPath", required=True)
parser.add_argument("--valid-images-path", dest="valImagesPath", default=None)
parser.add_argument("--valid-annotations-path", dest="valAnnotationsPath", default=None)

args = parser.parse_args()

print(args)
model_name = args.mdlType
if model_name == "vgg_unet":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=args.classNum, input_height=args.imgHeight, input_width=args.imgWidth)
elif model_name == "segnet":
    from keras_segmentation.models.segnet import segnet
    model = segnet(n_classes=args.classNum, input_height=args.imgHeight, input_width=args.imgWidth)
# elif model_name == "densenet121_unet_dt" or model_name == "densenet121_unet_dt_big_dataset":
#     from keras_segmentation.models.unet import densenet_unet
#     model = densenet_unet(n_classes=2, input_height=480, input_width=640)
else:
    raise Exception("UnknownModelError")

val_flag = True
if args.valImagesPath is None or args.valAnnotationsPath is None:
    val_flag = False
    print("validation path not stated, disabling validation.")


if not os.path.exists(args.mdlSavePath + "/checkpoints"):
    print("checkpoints save directory doesn't exists, creating...")
    os.mkdir(args.mdlSavePath + "/checkpoints")

model.train(
    train_images=args.trainImagesPath,
    train_annotations=args.trainAnnotationsPath,
    steps_per_epoch=args.stepsPerEpochs,
    validate=val_flag,
    val_steps_per_epoch=args.valStepsPerEpoch,
    val_images=args.valImagesPath,
    val_annotations=args.valAnnotationsPath,
    checkpoints_path=args.mdlSavePath + "/checkpoints/" + model_name,
    epochs=args.epochs
)

if not os.path.exists(args.mdlSavePath + "/weights"):
    print("weights save directory doesn't exists, creating...")
    os.mkdir(args.mdlSavePath + "/weights")
model.save_weights(args.mdlSavePath + "/weights/" + model_name + "_weights")
print("model saved successfully")
