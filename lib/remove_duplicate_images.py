import os
import cv2
import argparse

import numpy as np
import file_utils as fu
from imutils import paths


DIRECTORY = "/home/nightrider/calacademy-fish-id/datasets/image_classification/pcr/all_classes/acanthurus_triostegus/combined"
REMOVE = 0


def dhash(image, hashSize=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal
	# gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


if __name__ == "__main__":
	# grab the paths to all images in our input dataset directory and then initialize our hashes dictionary

	print("[INFO] computing image hashes...")
	imagePaths = fu.find_files(directory=DIRECTORY, extension=".jpg")
	hashes = {}

	# loop over our image paths
	for imagePath in imagePaths:
		# load the input image and compute the hash
		image = cv2.imread(imagePath)
		h = dhash(image)
		# grab all image paths with that hash, add the current image
		# path to it, and store the list back in the hashes dictionary
		p = hashes.get(h, [])
		p.append(imagePath)
		hashes[h] = p

	# count the duplicate sets for visibility into data
	num_duplicate_sets = 0
	for (h, hashedPaths) in hashes.items():
		if len(hashedPaths) > 1:
			num_duplicate_sets += 1
	print("Found {} duplicate sets".format(num_duplicate_sets))

	# loop over the image hashes
	for (h, hashedPaths) in hashes.items():
		# check to see if there is more than one image with the same hash
		if len(hashedPaths) > 1:
			# check to see if this is a dry run
			if REMOVE <= 0:
				# initialize a montage to store all images with the same
				# hash
				montage = None
				# loop over all image paths with the same hash
				for p in hashedPaths:
					# load the input image and resize it to a fixed width
					# and heightG
					image = cv2.imread(p)
					image = cv2.resize(image, (150, 150))
					# if our montage is None, initialize it
					if montage is None:
						montage = image
					# otherwise, horizontally stack the images
					else:
						montage = np.hstack([montage, image])
				# show the montage for the hash
				print("[INFO] hash: {}".format(h))
				cv2.imshow("Montage", montage)
				cv2.waitKey(0)
			# otherwise, we'll be removing the duplicate images
			else:
				# loop over all image paths with the same hash *except*
				# for the first image in the list (since we want to keep
				# one, and only one, of the duplicate images)
				for p in hashedPaths[1:]:
					print("Removing: {}".format(p))
					os.remove(p)