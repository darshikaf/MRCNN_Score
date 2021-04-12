import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def add_class(source, class_id, class_name):
    class_info = [{"source": "", "id": 0, "name": "BG"}]
    assert "." not in source, "Source name cannot contain a dot"
    # Does the class exist already?
    for info in class_info:
        if info['source'] == source and info["id"] == class_id:
            # source.class_id combination already available, skip
            return
    # Add the class
    class_info.append({
        "source": source,
        "id": class_id,
        "name": class_name,
    })

def load_image(file_path):
    image = skimage.io.imread(file_path)
    if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
    if image.shape[-1] == 4:
            image = image[..., :3]
    return image

