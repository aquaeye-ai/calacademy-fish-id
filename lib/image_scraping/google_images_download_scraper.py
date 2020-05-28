import errno
import warnings

# importing google_images_download module
from google_images_download import google_images_download

# creating object
response = google_images_download.googleimagesdownload()

search_queries = \
    [
        'caesio teres'
    ]


def downloadimages(query):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {
        "keywords": query,
        "format": "jpg",
        "limit": 30,
        "print_urls": True,
        "size": "large",
        "related_images": None,
        "output_directory": '/home/nightrider/calacademy-fish-id/datasets/image_classification/two_class',
        "image_directory": 'caesio teres',
        "aspect_ratio": "panoramic"
    }
    try:
        response.download(arguments)

        # Handling File NotFound Error
    except OSError as e:
        if e.errno == errno.ENOENT:
            arguments = {
                "keywords": query,
                "format": "jpg",
                "limit": 1,
                "print_urls": True,
                "size": "large",
                "related_images": None,
                "output_directory": '/home/nightrider/calacademy-fish-id/datasets/image_classification/two_class',
                "image_directory": 'caesio teres'
            }

            # Providing arguments for the searched query
            try:
                # Downloading the photos based
                # on the given arguments
                response.download(arguments)
            except:
                pass
        else:
            raise


if __name__ == "__main__":
    # Google changed DOM that breaks this python package: https://github.com/hardikvasa/google-images-download/pull/298
    # https://stackoverflow.com/questions/60370799/google-image-download-with-python-cannot-download-images
    warnings.warn("deprecated", DeprecationWarning)

    # Driver Code
    for query in search_queries:
        downloadimages(query)
        print()