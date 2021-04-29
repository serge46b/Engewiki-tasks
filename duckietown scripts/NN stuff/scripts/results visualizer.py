import matplotlib.pyplot as plt
from IPython.display import Image
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
    raise Exception("entered unknown model name")

path = "C:/Users/Serg/Desktop/some_files/engewiki duckietown NN course/keras segmentation data/"

model.load_weights(path + "tmp/" + model_name + "/weights/" + model_name + "_weights")

# Sky, Building, Pole, Road, Pavement, Tree, SingSymbol, Fence, Car, Pedestrian, Bicyclist
# Other, RoadMark
classes = input("enter model classes: ").split(", ")

# dataset1/images_prepped_test/0016E5_07965.png
# dataset2/test/screenshot.1.png
test_path = path + input("enter path to test image: ")
results_path = path + "tmp/" + model_name + "/results/"

out = model.predict_segmentation(
    inp=test_path,
    out_fname=results_path + model_name + "_out.png"
)

Image(results_path + model_name + "_out.png")

out_segment = model.predict_segmentation(
    inp=test_path,
    out_fname=results_path + model_name + "_out_segment.png", overlay_img=True, show_legends=True,
    class_names=classes

)

Image(results_path + model_name + "_out_segment.png")

print("resulting images successfully saved")
