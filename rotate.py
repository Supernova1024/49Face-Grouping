###################################################################################
## For the production, make belows function as comment
##   print_rect(image_to_check, unknown_image, face_location, rotate_id) //line 79
##   image = draw_rect  //line 168
##   faces = collect_faces  //line 244
##   reate a montage using 96x96 "tiles" with 5 rows and 5 columns //line 246
####################################################################################   

from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import sys
import itertools
import PIL.Image
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from imutils import build_montages
import matplotlib.pyplot as plt
import time


CLUSTERING_RESULT_PATH = os.getcwd()

data = []

def current_milli_time():
    return round(time.time() * 1000)


def print_result(filename, location):
    top, right, bottom, left = location
    print("{},{},{},{},{}".format(filename, top, right, bottom, left))


def print_rect(image_to_check, unknown_image, face_location, rotate_id):
    if rotate_id == 0:
        (top, right, bottom, left) = face_location
        image = cv2.rectangle(unknown_image, (left, top), (right, bottom),(0, 255, 0), 5)
        path = "p_result/" + image_to_check.split("/")[1]
        cv2.imwrite(path, image)
    elif rotate_id == 1:
        (top, right, bottom, left) = face_location
        image = cv2.rectangle(unknown_image, (left, top), (right, bottom),(0, 255, 0), 5)
        image = img_rotate(image, 2)
        path = "p_result/" + image_to_check.split("/")[1]
        cv2.imwrite(path, image)
    elif rotate_id == 2:
        (top, right, bottom, left) = face_location
        image = cv2.rectangle(unknown_image, (left, top), (right, bottom),(0, 255, 0), 5)
        image = img_rotate(image, 1)
        path = "p_result/" + image_to_check.split("/")[1]
        cv2.imwrite(path, image)
    else:
        (top, right, bottom, left) = face_location
        image = cv2.rectangle(unknown_image, (left, top), (right, bottom),(0, 255, 0), 5)
        image = img_rotate(image, 3)
        path = "p_result/" + image_to_check.split("/")[1]
        cv2.imwrite(path, image)


def img_rotate(img, i):
   if i == 1:
      image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
   elif i == 2:
      image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
   else:
      image = cv2.rotate(img, cv2.ROTATE_180)
   return image

def push_data(face_locations, unknown_image, image_to_check, rotate_id):
    for face_location in face_locations:
        print_result(image_to_check, face_location)

        ## call print function
        # print_rect(image_to_check, unknown_image, face_location, rotate_id)

    encodings = face_recognition.face_encodings(unknown_image, face_locations)


    d = [{"imagePath": image_to_check, "loc": box, "encoding": enc, "rotate_id": rotate_id}
        for (box, enc) in zip(face_locations, encodings)]
    
    data.extend(d)


def test_image(image_to_check, model, upsample):
    unknown_image = face_recognition.load_image_file(image_to_check)
    face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=upsample, model=model)
    print("====face_locations==", face_locations)
    if len(face_locations) != 0:
        push_data(face_locations, unknown_image, image_to_check, 0)
    else:
        rotate_id = 1
        flag = "undetected"
        while rotate_id < 4:
            unknown_image = img_rotate(unknown_image, rotate_id)
            face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=upsample, model=model)
            if len(face_locations) != 0:
                push_data(face_locations, unknown_image, image_to_check, rotate_id)
                flag = "detected"
                break;
            rotate_id += 1

        if flag == "undetected":
            img_id = image_to_check.split("/")[1].split(".")[0]
            move_image(unknown_image, img_id, "no_person", 4, "location")


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, number_of_cpus, model, upsample):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(model),
        itertools.repeat(upsample),
    )

    pool.starmap(test_image, function_parameters)

def draw_rect(image, rotate_id, location):
    if rotate_id == 0:
        (top, right, bottom, left) = location
        image = cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 255), 10)

    elif rotate_id == 1:
        (top, right, bottom, left) = location
        image = img_rotate(image, 1)
        image = cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 255), 10)
        image = img_rotate(image, 2)

    elif rotate_id == 2:
        (top, right, bottom, left) = location
        image = img_rotate(image, 2)
        image = cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 255), 10)
        image = img_rotate(image, 1)

    elif rotate_id == 3:
        (top, right, bottom, left) = location
        image = img_rotate(image, 3)
        image = cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 255), 10)
        image = img_rotate(image, 3)

    else:
        image = image

    return image

def move_image(image,id,labelID, rotate_id, location):
    
    # image = draw_rect(image, rotate_id, location)

    path = CLUSTERING_RESULT_PATH+'/label'+str(labelID)
    # os.path.exists() method in Python is used to check whether the specified path exists or not.
    # os.mkdir() method in Python is used to create a directory named path with the specified numeric mode.
    if os.path.exists(path) == False:
        os.mkdir(path)

    filename = str(id) +'.jpg'
    cv2.imwrite(os.path.join(path , filename), image)
    
    return


def collect_faces(location, rotate_id, image, faces):
    (top, right, bottom, left) = location
    if rotate_id == 0:
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        faces.append(face)
    elif rotate_id == 1:
        image = img_rotate(image, 1)
        face = image[top:bottom, left:right]
        face = img_rotate(face, 2)
        face = cv2.resize(face, (96, 96))
        faces.append(face)
    elif rotate_id == 2:
        image = img_rotate(image, 2)
        face = image[top:bottom, left:right]
        face = img_rotate(face, 1)
        face = cv2.resize(face, (96, 96))
        faces.append(face)
    else:
        image = img_rotate(image, 3)
        face = image[top:bottom, left:right]
        face = img_rotate(face, 3)
        face = cv2.resize(face, (96, 96))
        faces.append(face)
    return faces


def faces_grouping(data, cpus):
    encodings = [d["encoding"] for d in data]

    # cluster the embeddings
    print("[INFO] clustering...")

    # creating DBSCAN object for clustering the encodings with the metric "euclidean"
    clt = DBSCAN(eps=0.31, metric="euclidean", n_jobs=cpus, min_samples=1)
    clt.fit(encodings)
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    n_clusters_ = len(set(labelIDs)) - (1 if -1 in labelIDs else 0)
    n_noise_ = list(labelIDs).count(-1)
    print("n_clusters_:", n_clusters_)
    print("n_noise_:", n_noise_)

    # ======loop over the unique face integers
    for labelID in labelIDs:
        print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]

        # initialize the list of faces to include in the montage
        faces = []

        # loop over the sampled indexes
        if len(idxs) > 0:
            for i in idxs:
                # load the input image and extract the face ROI
                image = cv2.imread(data[i]["imagePath"])
                move_image(image,i,labelID,data[i]["rotate_id"], data[i]["loc"])

                # force resize the face ROI to 96mx96 and then add it to the
                # faces montage list
                # faces = collect_faces(data[i]["loc"], data[i]["rotate_id"], image, faces)

        # # # ======create a montage using 96x96 "tiles" with 5 rows and 5 columns
        # montage = build_montages(faces, (96, 96), (7, 7))[0]
        # # show the output montage
        # title = "Face ID #{}".format(labelID)
        # title = "Unknown Faces" if labelID == -1 else title
        # cv2.imwrite(os.path.join(CLUSTERING_RESULT_PATH, title+'.jpg'), montage)

@click.command()
@click.argument('image_to_check')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel. -1 means "use all in system"')
@click.option('--model', default="hog", help='Which face detection model to use. Options are "hog" or "cnn".')
@click.option('--upsample', default=0, help='How many times to upsample the image looking for faces. Higher numbers find smaller faces.')
def main(image_to_check, cpus, model, upsample):

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, model, upsample) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), cpus, model, upsample)
    else:
        test_image(image_to_check, model, upsample)

    if len(data) != 0:
        faces_grouping(data, cpus)

if __name__ == "__main__":
    main()
