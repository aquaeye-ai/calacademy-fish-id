"""
Script to scrape google images.
Adapted from: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
"""

import os
import cv2
import lib.file_utils as fu
import requests


URLS = "/home/nightrider/Downloads/urls.txt"
OUTPUT = "/home/nightrider/calacademy-fish-id/datasets/image_classification/two_class/caesio_teres"

def download_images_for_urls(urls=None, dst_dir=None):
    rows = open(urls).read().strip().split("\n")
    total = 0

    # loop the URLs
    for url in rows:
        try:
            # try to download the image
            r = requests.get(url, timeout=60)
            # save the image to disk
            p = os.path.sep.join([dst_dir, "{}.jpg".format(str(total).zfill(8))])
            f = open(p, "wb")
            f.write(r.content)
            f.close()
            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1
        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))

def remove_corrupted_images(src_dr=None):
    # loop over the image paths we just downloaded and remove ones that can't be opened
    for imagePath in fu.find_files(directory=src_dr, extension=".jpg"):
        # initialize if the image should be deleted or not
        delete = False
        # try to load the image
        try:
            image = cv2.imread(imagePath)
            # if the image is `None` then we could not properly load it
            # from disk, so delete it
            if image is None:
                delete = True
        # if OpenCV cannot load the image then the image is likely
        # corrupt so we should delete it
        except:
            print("Except")
            delete = True
        # check to see if the image should be deleted
        if delete:
            print("[INFO] deleting {}".format(imagePath))
            os.remove(imagePath)

if __name__ == "__main__":
    # download images using scraped urls
    download_images_for_urls(urls=URLS, dst_dir=OUTPUT)

    # remove corrupted images
    remove_corrupted_images(src_dr=OUTPUT)