# import matplotlib.pyplot as plt
from IPython.display import Image
import os
model_name = input("Enter model name: ")
if model_name == "vgg_unet":
    from keras_segmentation.models.unet import vgg_unet
    model = vgg_unet(n_classes=50, input_height=320, input_width=640)
elif model_name.startswith("vgg_unet"):
    from keras_segmentation.models.unet import vgg_unet
    if "mdt" in model_name:
        model = vgg_unet(n_classes=4, input_height=480, input_width=640)
    else:
        model = vgg_unet(n_classes=2, input_height=480, input_width=640)
elif model_name.startswith("segnet"):
    from keras_segmentation.models.segnet import segnet
    if "mdt" in model_name:
        model = segnet(n_classes=4, input_height=480, input_width=640)
    else:
        model = segnet(n_classes=2, input_height=480, input_width=640)
# elif model_name == "densenet121_unet_dt" or model_name == "densenet121_unet_dt_big_dataset":
#     from keras_segmentation.models.unet import densenet_unet
#     model = densenet_unet(n_classes=2, input_height=480, input_width=640)
else:
    raise Exception("UnknownModelError")

#path = "C:/Serge/Desktop/files/project files/engewiki duckietown NN course/keras segmentation data/"
#path = "C:/Users/Serg/Desktop/some_files/engewiki duckietown NN course/keras segmentation data/"
path = "H:/some_files/engewiki duckietown NN course/keras segmentation data/"

model.load_weights(path + "tmp/" + model_name + "/weights/" + model_name + "_weights")

# Sky, Building, Pole, Road, Pavement, Tree, SingSymbol, Fence, Car, Pedestrian, Bicyclist
# Other, RoadMark
# Other, RoadMark, YellowRoadMark, CrossroadMarks
classes = input("enter model classes: ").split(", ")

# dataset1/images_prepped_test/0016E5_07965.png
# dataset_dt/test/screenshot.1.png
# dataset_dt2/test/img.1.png
# dataset_dt3/test/img.1.png
# dataset_dt4/test/img.1.png
# dataset_mdt/test/
test_path = path + input("enter path to test image: ")
save_path = path + "tmp/" + model_name + "/results/"
if not os.path.exists(save_path):
    print("save directory doesn't exists, creating...")
    os.mkdir(save_path)
results_name = save_path + "out_" + input("enter resulting image name: ") + "_"

out = model.predict_segmentation(
    inp=test_path,
    out_fname=results_name + model_name + ".png",
    colors=[(0, 0, 0), (255, 255, 255), (0, 255, 255), (0, 0, 255)]
)

Image(results_name + model_name + ".png")

out_segment = model.predict_segmentation(
    inp=test_path,
    out_fname=results_name + model_name + "_segment.png", overlay_img=True, show_legends=True,
    class_names=classes,
    colors=[(0, 0, 0), (255, 255, 255), (0, 255, 255), (0, 0, 255)]

)

Image(results_name + model_name + "_segment.png")

print("resulting images successfully saved")
