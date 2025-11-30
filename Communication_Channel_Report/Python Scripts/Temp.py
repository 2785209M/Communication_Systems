from PIL import Image

def run_length_encoding(image_array):
    encoded_image = ""

    for row in image_array:
        current_pixel = row[0]
        run_length = 1

        for pixel in row[1:]:
            if pixel == current_pixel:
                run_length += 1
            else:
                encoded_image += f"{run_length}:{current_pixel} "
                current_pixel = pixel
                run_length = 1

        encoded_image += f"{run_length}:{current_pixel}\n"

    return encoded_image


def decode_run_length(encoded_text):
    image_array = []

    lines = encoded_text.strip().split("\n")

    for line in lines:
        row = []
        tokens = line.split()

        for token in tokens:
            run, pixel = token.split(":")
            run = int(run)
            pixel = int(pixel)
            row.extend([pixel] * run)

        image_array.append(row)

    return image_array



def load_png_as_array(path, binary_threshold=None):
    img = Image.open(path).convert("L")  # grayscale (0–255)

    pixels = list(img.getdata())
    width, height = img.size
    pixel_array = [pixels[i * width:(i + 1) * width] for i in range(height)]

    # Convert to 0/1 binary if chosen
    if binary_threshold is not None:
        pixel_array = [[1 if p > binary_threshold else 0 for p in row] for row in pixel_array]

    return pixel_array


def save_array_as_png(pixel_array, path):
    """
    Saves a 2D array of 0/1 (or 0–255 grayscale) pixels as a PNG image.
    """

    height = len(pixel_array)
    width = len(pixel_array[0])

    # Convert 0/1 binary back to grayscale
    flattened = []
    for row in pixel_array:
        for p in row:
            flattened.append(255 if p == 1 else 0)

    img = Image.new("L", (width, height))
    img.putdata(flattened)
    img.save(path)
    print(f"Image saved as {path}")


# -----------------------------
# Example usage
# -----------------------------

# 1. Load PNG → array
image_array = load_png_as_array(
    "/home/james/University/Communication_Systems/Communication_Channel_Report/image.jpg",
    binary_threshold=128
)

# 2. Encode image → text
encoded_image = run_length_encoding(image_array)
print(encoded_image)

# 3. Decode text → array
decoded_array = decode_run_length(encoded_image)

# 4. Save decoded array → PNG
save_array_as_png(decoded_array, "decoded_output.png")