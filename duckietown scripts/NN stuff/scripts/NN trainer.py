import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_name = input("Enter model name: ")
if model_name == "vgg_unet":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=50, input_height=320, input_width=640)
elif model_name == "vgg_unet_dt" or model_name == "vgg_unet_dt_mid_dataset" or model_name == "vgg_unet_dt_big_dataset" \
        or model_name == "vgg_unet_dt_mid_dataset_dt4" or model_name == "vgg_unet_dt_big_dataset_dt5":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=2, input_height=480, input_width=640)
elif model_name == "segnet_dt" or model_name == "segnet_dt_mid_dataset" or model_name == "segnet_dt_big_dataset"\
        or model_name == "segnet_dt_mid_dataset_dt4" or model_name == "segnet_dt_big_dataset_dt5":
    from keras_segmentation.models.segnet import segnet
    model = segnet(n_classes=2, input_height=480, input_width=640)
elif model_name == "densenet121_unet_dt" or model_name == "densenet121_unet_dt_big_dataset":
    from keras_segmentation.models.unet import densenet_unet
    model = densenet_unet(n_classes=2, input_height=480, input_width=640)
else:
    raise Exception("UnknownModelError")


path = "H:/some_files/engewiki duckietown NN course/keras segmentation data/"
# dataset1/images_prepped_train/
# dataset_dt/images/
# dataset_dt2/images/
# dataset_dt3/images/
# dataset_dt4/images/
# dataset_dt5/images/
train_images_path = path + input("enter train images path: ")
# dataset1/annotations_prepped_train/
# dataset_dt/masks/
# dataset_dt2/masks/
# dataset_dt3/masks/
# dataset_dt4/masks/
# dataset_dt5/masks/
train_masks_path = path + input("enter train annotations path: ")

model.train(
    train_images=train_images_path,
    train_annotations=train_masks_path,
    checkpoints_path=path + "tmp/" + model_name + "/checkpoints/" + model_name,
    # epochs=5
    epochs=2
)

# model.save(path + "tmp/" + model_name)
model.save_weights(path + "tmp/" + model_name + "/weights/" + model_name + "_weights")
