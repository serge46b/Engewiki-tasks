model_name = input("Enter model name: ")
if model_name == "vgg_unet":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=50, input_height=320, input_width=640)
elif model_name == "vgg_unet_dt":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=2, input_height=480, input_width=640)
elif model_name == "segnet_dt":
    from keras_segmentation.models.segnet import segnet
    model = segnet(n_classes=2, input_height=480, input_width=640)
else:
    raise Exception("UnknownModelError")

path = "C:/Users/Serg/Desktop/some_files/engewiki duckietown NN course/keras segmentation data/"
# dataset1/images_prepped_train/
# dataset2/images/
train_images_path = path + input("enter train images path: ")
# dataset1/annotations_prepped_train/
# dataset2/masks/
train_masks_path = path + input("enter train annotations path: ")

model.train(
    train_images=train_images_path,
    train_annotations=train_masks_path,
    checkpoints_path=path + "tmp/" + model_name + "/checkpoints/" + model_name, epochs=5
)

# model.save(path + "tmp/" + model_name)
model.save_weights(path + "tmp/" + model_name + "/weights/" + model_name + "_weights")
