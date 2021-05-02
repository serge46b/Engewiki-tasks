import matplotlib.pyplot as plt
from IPython.display import Image
model_name = input("Enter model name: ")
if model_name == "vgg_unet":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=50, input_height=320, input_width=640)
elif model_name == "vgg_unet_dt" or model_name == "vgg_unet_dt_mid_dataset" or model_name == "vgg_unet_dt_big_dataset"\
        or model_name == "vgg_unet_dt_mid_dataset_dt4" or model_name == "vgg_unet_dt_big_dataset_dt5":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=2, input_height=480, input_width=640)
elif model_name == "segnet_dt" or model_name == "segnet_dt_mid_dataset" or model_name == "segnet_dt_big_dataset" \
        or model_name == "segnet_dt_mid_dataset_dt4" or model_name == "segnet_dt_big_dataset_dt5":
    from keras_segmentation.models.segnet import segnet
    model = segnet(n_classes=2, input_height=480, input_width=640)
else:
    raise Exception("UnknownModelError")

path = "A:/Serge/Desktop/files/project files/engewiki duckietown NN course/keras segmentation data/"

model.load_weights(path + "tmp/" + model_name + "/weights/" + model_name + "_weights")

# Sky, Building, Pole, Road, Pavement, Tree, SingSymbol, Fence, Car, Pedestrian, Bicyclist
# Other, RoadMark
classes = input("enter model classes: ").split(", ")

# dataset1/images_prepped_test/0016E5_07965.png
# dataset_dt/test/screenshot.1.png
# dataset_dt2/test/img.1.png
# dataset_dt3/test/img.1.png
# dataset_dt4/test/img.1.png
test_path = path + input("enter path to test image: ")
results_name = path + "tmp/" + model_name + "/results/out_" + input("enter resulting image name: ")

out = model.predict_segmentation(
    inp=test_path,
    out_fname=results_name + model_name + ".png"
)

Image(results_name + model_name + ".png")

out_segment = model.predict_segmentation(
    inp=test_path,
    out_fname=results_name + model_name + "_segment.png", overlay_img=True, show_legends=True,
    class_names=classes

)

Image(results_name + model_name + "_segment.png")

print("resulting images successfully saved")
