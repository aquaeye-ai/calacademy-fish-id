"""
Script to scrape google images.
Adapted from: https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

One must first manually google search for images, open the js console in the browser and then paste the following code
snippets to download a 'urls.txt' file containing image links before this script can be run successfully:

/**
 * simulate a right-click event so we can grab the image URL using the
 * context menu alleviating the need to navigate to another page
 *
 * attributed to @jmiserez: http://pyimg.co/9qe7y
 *
 * @param   {object}  element  DOM Element
 *
 * @return  {void}
 */
function simulateRightClick( element ) {
    var event1 = new MouseEvent( 'mousedown', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 2,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event1 );
    var event2 = new MouseEvent( 'mouseup', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 0,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event2 );
    var event3 = new MouseEvent( 'contextmenu', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 0,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event3 );
}

/**
 * grabs a URL Parameter from a query string because Google Images
 * stores the full image URL in a query parameter
 *
 * @param   {string}  queryString  The Query String
 * @param   {string}  key          The key to grab a value for
 *
 * @return  {string}               value
 */
function getURLParam( queryString, key ) {
    var vars = queryString.replace( /^\?/, '' ).split( '&' );
    for ( let i = 0; i < vars.length; i++ ) {
        let pair = vars[ i ].split( '=' );
        if ( pair[0] == key ) {
            return pair[1];
        }
    }
    return false;
}

/**
 * Generate and automatically download a txt file from the URL contents
 *
 * @param   {string}  contents  The contents to download
 *
 * @return  {void}
 */
function createDownload( contents ) {
    var hiddenElement = document.createElement( 'a' );
    hiddenElement.href = 'data:attachment/text,' + encodeURI( contents );
    hiddenElement.target = '_blank';
    hiddenElement.download = 'urls.txt';
    hiddenElement.click();
}

/**
 * grab all URLs va a Promise that resolves once all URLs have been
 * acquired
 *
 * @return  {object}  Promise object
 */
function grabUrls() {
    var urls = [];
    return new Promise( function( resolve, reject ) {
        var count = document.querySelectorAll(
        	'.isv-r a:first-of-type' ).length,
            index = 0;
        Array.prototype.forEach.call( document.querySelectorAll(
        	'.isv-r a:first-of-type' ), function( element ) {
            // using the right click menu Google will generate the
            // full-size URL; won't work in Internet Explorer
            // (http://pyimg.co/byukr)
            simulateRightClick( element.querySelector( ':scope img' ) );
            // Wait for it to appear on the <a> element
            var interval = setInterval( function() {
                if ( element.href.trim() !== '' ) {
                    clearInterval( interval );
                    // extract the full-size version of the image
                    let googleUrl = element.href.replace( /.*(\?)/, '$1' ),
                        fullImageUrl = decodeURIComponent(
                        	getURLParam( googleUrl, 'imgurl' ) );
                    if ( fullImageUrl !== 'false' ) {
                        urls.push( fullImageUrl );
                    }
                    // sometimes the URL returns a "false" string and
                    // we still want to count those so our Promise
                    // resolves
                    index++;
                    if ( index == ( count - 1 ) ) {
                        resolve( urls );
                    }
                }
            }, 10 );
        } );
    } );
}

/**
 * Call the main function to grab the URLs and initiate the download
 */
grabUrls().then( function( urls ) {
    urls = urls.join( '\n' );
    createDownload( urls );
} );

"""

import os
import cv2
import lib.file_utils as fu
import requests


URLS = "/home/nightrider/Downloads/urls.txt"
OUTPUT = "/home/nightrider/calacademy-fish-id/datasets/image_classification/reef_lagoon/scraped_web/master/trachinotus_mookalee/google"

def download_images_for_urls(urls=None, dst_dir=None):
    rows = open(urls).read().strip().split("\n")
    total = 0

    # loop the URLs
    for url in rows:
        try:
            print(url)
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
            print("[INFO] error downloading {}...skipping".format(url))

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
    # make sure directory is valid
    fu.init_directory(directory=OUTPUT)

    # download images using scraped urls
    download_images_for_urls(urls=URLS, dst_dir=OUTPUT)

    # remove corrupted images
    remove_corrupted_images(src_dr=OUTPUT)

    # remove downloaded urls.txt file
    os.remove(URLS)