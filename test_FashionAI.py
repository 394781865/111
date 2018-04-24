import os
import datetime
import numpy as np
import model as modellib
import FashionAI
import skimage.io


DATA_DIR = "./data"

annotations_path = {
    "test": os.path.join(DATA_DIR, "test/test.csv"),
}

class InferenceConfig(FashionAI.FashionConfig):
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 56

test_model = "test"
check_model_num = "0019"
WANT_SAVE_CSV = True
head = ["image_id","image_category","neckline_left","neckline_right","center_front",
       "shoulder_left","shoulder_right","armpit_left","armpit_right","waistline_left","waistline_right",
       "cuff_left_in","cuff_left_out","cuff_right_in","cuff_right_out","top_hem_left","top_hem_right",
       "waistband_left","waistband_right","hemline_left","hemline_right","crotch","bottom_left_in",
       "bottom_left_out","bottom_right_in","bottom_right_out"]

result_csv = [head]
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# set config
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_"+check_model_num+".h5")
test_annotations_list = np.loadtxt(annotations_path[test_model], skiprows=1, delimiter=',', dtype=bytes).astype(str)
inference_config = InferenceConfig()
inference_config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inference_config)
assert COCO_MODEL_PATH != "", "Provide path to trained weights"

# Load weights trained on COCO_MODEL_PATH
model.load_weights(COCO_MODEL_PATH)
print("Loading weights from ", COCO_MODEL_PATH)

count = 0
all_valid = []
for m in range(len(test_annotations_list)):
    row_annotation = test_annotations_list[m]
    img_path = os.path.join(DATA_DIR, "test", row_annotation[0])
    clothes_type = row_annotation[0].split("/")[1]
    original_image = skimage.io.imread(img_path)
    original_image_copy = original_image.copy()
    # predict the keypoints of the original_image
    results = model.detect_keypoint([original_image])
    r = results[0]
    if WANT_SAVE_CSV:
        row = ["-1_-1_-1" for i in range(26)]
        row[0] = row_annotation[0]
        row[1] = row_annotation[1]
        all_keypoints = r['keypoints'][0]
        for i in FashionAI.clothes_index[clothes_type]:
            str_one = str(all_keypoints[i-2][0]) + "_" + str(all_keypoints[i-2][1]) + "_1"
            row[i] = str_one
        row = np.array(row).astype(str)
        result_csv.append(row)

    count += 1
    if count % 10 == 0:
        print(str(count) + "/" + str(len(test_annotations_list)))

if WANT_SAVE_CSV:
    result_csv = np.array(result_csv)
    np.savetxt("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv", result_csv, fmt='%s', delimiter=',')