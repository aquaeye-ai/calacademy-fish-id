"""
Computes similarities of all images in a directory given a hashSize.
hashSize = 8 has been found to find only identical images.
hashSize = 4 has been found to give near-identical images.

If AUTO_REMOVE != 0, then the script will prompt user to for a set of found, identical images.
The prompt will display the found duplicates in a window, left-to-right, in descending order of dimensions.
Pressing 'd' will remove all but left-most image.
Pressing 'm' will move all but left-most image to a separate folder called "potential_duplicates".
Pressing any other key will ignore the current set and move on to the next set.

Adapted from: https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/
"""


import os
import cv2
import shutil

import numpy as np
import file_utils as fu

BASE_DIRECTORY = "/home/nightrider/calacademy-fish-id/datasets/object_detection/reef_lagoon/stills/full/temp/scraped/stingray"
SOURCE_DIRECTORY = os.path.join(BASE_DIRECTORY, "images")
POTENTIAL_DUPLICATES_DIRECTORY = os.path.join(BASE_DIRECTORY, "potential_duplicates")
AUTO_REMOVE = 0



def dhash(image, hashSize=4):#8):
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
	imagePaths = fu.find_files(directory=SOURCE_DIRECTORY, extension=".jpg")
	hashes = {}

	# loop over our image paths
	for imagePath in imagePaths:
		# load the input image and compute the hash
		image = cv2.imread(imagePath)
		h = dhash(image)
		# grab all image paths with that hash, add the current image
		# path to it, and store the list back in the hashes dictionary
		p = hashes.get(h, [])

		# sort the image paths by image size -> largest to smallest
		# this makes it easy to retain the high resolution images when removing duplicates later
		new_img_size = image.shape[0]*image.shape[1]
		insert_idx = len(p)
		for idx, img_path in enumerate(p):
			img = cv2.imread(img_path)
			img_size = img.shape[0]*img.shape[1]

			if new_img_size > img_size:
				insert_idx = idx
				break

		# p.append(imagePath)
		p.insert(insert_idx, imagePath)
		hashes[h] = p

	# count the duplicate sets for visibility into data
	num_duplicate_sets = 0
	for (h, hashedPaths) in hashes.items():
		if len(hashedPaths) > 1:
			num_duplicate_sets += 1
	print("[INFO] Found {} duplicate sets".format(num_duplicate_sets))

	# loop over the image hashes
	for (h, hashedPaths) in hashes.items():
		# check to see if there is more than one image with the same hash
		if len(hashedPaths) > 1:
			# check to see if this is a dry run
			if AUTO_REMOVE <= 0:
				# initialize a montage to store all images with the same
				# hash
				montage = None
				# loop over all image paths with the same hash
				for p in hashedPaths:
					# load the input image and resize it to a fixed width
					# and heightG
					image = cv2.imread(p)
					print("[INFO] path: {}; size: {}".format(p, image.shape[0:2]))
					image = cv2.resize(image, (512, 512))
					# if our montage is None, initialize it
					if montage is None:
						montage = image
					# otherwise, horizontally stack the images
					else:
						montage = np.hstack([montage, image])
				# show the montage for the hash
				print("[INFO] hash: {}".format(h))
				cv2.imshow("Press: 'd' to remove duplicates; 'm' to move all but left-most image to potential_duplicates directory; Any other key to continue", montage)

				# wait for user defined action before proceeding
				k = cv2.waitKey(0)
				if k == ord('d'): # remove duplicates
					# loop over all image paths with the same hash *except*
					# for the first image in the list (since we want to keep
					# one, and only one, of the duplicate images)
					for p in hashedPaths[1:]:
						print("Removing: {}".format(p))
						os.remove(p)
				elif k == ord('m'): # move to a folder for further analysis
					# loop over all image paths with the same hash *except*
					# for the first image in the list (since we want to keep
					# one, and only one, of the duplicate images)
					for p in hashedPaths[1:]:
						print("Moving: {}".format(p))

						# initialize potential duplicates directory if it doesn't exist
						fu.init_directory(directory=POTENTIAL_DUPLICATES_DIRECTORY)

						# move file
						basename = os.path.basename(p)
						shutil.move(p, os.path.join(POTENTIAL_DUPLICATES_DIRECTORY, basename))

			# otherwise, we'll be removing the duplicate images
			else:
				# loop over all image paths with the same hash *except*
				# for the first image in the list (since we want to keep
				# one, and only one, of the duplicate images)
				for p in hashedPaths[1:]:
					print("Removing: {}".format(p))
					os.remove(p)