class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        results["filename"] = None
        results["ori_filename"] = None
        results["img_shape"] = results["img"].shape
        results["ori_shape"] = results["img"].shape
        return results
