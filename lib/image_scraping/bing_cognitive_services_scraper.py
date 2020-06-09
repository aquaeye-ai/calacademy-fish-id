# -*- coding: utf-8 -*-

"""
Script to scrape bing images using Microsoft's Cognitive Services API.
Adapted from: https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/
"""

from requests import exceptions

import cv2
import os
import errno
import requests

import lib.file_utils as fu


API_KEY = "e462070da4fd402a92a108160420c7f3"
# make sufficiently large, 2000 is recommended as ideal for object detection problems and we apply that advice to the
# image classification problem
MAX_RESULTS = 2000
GROUP_SIZE = 50

# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
# URL = "https://westcentralus.api.cognitive.microsoft.com/vision"

# when attempting to download images from the web both the Python
# programming language and the requests library have a number of
# exceptions that can be thrown so let's build a list of them now
# so we can filter on them
EXCEPTIONS = set([IOError, errno.ENOENT, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError,
                  exceptions.Timeout, exceptions.SSLError])

# QUERY = "Chromis ternatensis OR Ternate Chromis"
# OUTPUT = "/home/nightrider/calacademy-fish-id/datasets/image_classification/pcr/three_class/chromis_ternatensis/bing"

QUERIES = {
    "naso_unicornis": "Naso unicornis OR Blue-spine Unicornfish"
}

if __name__ == "__main__":
    # loop through our dictionary of species/queries, scrape images for each pair and store the resulting images
    for species, query in QUERIES.items():
        # create output directory
        output_parent_dir = "/home/nightrider/calacademy-fish-id/datasets/image_classification/pcr/three_class/"
        output_dir = os.path.join(output_parent_dir, species, "bing")
        fu.init_directory(directory=output_dir)

        # store the search term in a convenience variable then set the
        # headers and search parameters
        term = query
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        params = {"q": term, "offset": 0, "count": GROUP_SIZE}
        # make the search
        print("[INFO] searching Bing API for '{}'".format(term))
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()
        # grab the results from the search, including the total number of
        # estimated results returned by the Bing API
        results = search.json()
        estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
        print("[INFO] {} total results for '{}'".format(estNumResults,
                                                        term))
        # initialize the total number of images downloaded thus far
        total = 0

        # loop over the estimated number of results in `GROUP_SIZE` groups
        for offset in range(0, estNumResults, GROUP_SIZE):
            # update the search parameters using the current offset, then
            # make the request to fetch the results
            print("[INFO] making request for group {}-{} of {}...".format(
                offset, offset + GROUP_SIZE, estNumResults))
            params["offset"] = offset
            search = requests.get(URL, headers=headers, params=params)
            search.raise_for_status()
            results = search.json()
            print("[INFO] saving images for group {}-{} of {}...".format(
                offset, offset + GROUP_SIZE, estNumResults))

            # loop over the results
            for v in results["value"]:
                # try to download the image
                try:
                    # make a request to download the image
                    print("[INFO] fetching: {}".format(v["contentUrl"]))
                    r = requests.get(v["contentUrl"], timeout=30)
                    # build the path to the output image
                    ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                    p = os.path.sep.join([output_dir, "{}{}".format(
                        str(total).zfill(8), ext)])
                    # write the image to disk
                    f = open(p, "wb")
                    f.write(r.content)
                    f.close()
                # catch any errors that would not unable us to download the
                # image
                except Exception as e:
                    # check to see if our exception is in our list of
                    # exceptions to check for
                    if type(e) in EXCEPTIONS:
                        print("[INFO] skipping: {}".format(v["contentUrl"]))
                        continue

                # try to load the image from disk
                image = cv2.imread(p)
                # if the image is `None` then we could not properly load the
                # image from disk (so it should be ignored)
                if image is None and os.path.isfile(p):
                    print("[INFO] deleting: {}".format(p))
                    os.remove(p)
                    continue
                # update the counter
                total += 1