from pathlib import Path

import pytesseract as pt
from PIL import Image

# NOTE: You must install Tesseract OCR on Windows before using this script.
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# After installation, ensure Tesseract is added to the system PATH or specify the full path below.


def main():
    """Extracts text from an image using Tesseract OCR."""

    # User-defined settings (Easily editable at the top)
    tesseract_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )  # Set Tesseract path
    image_path = Path("test_image.png")  # Set image path

    # Set the Tesseract executable path
    pt.pytesseract.tesseract_cmd = str(tesseract_path)

    # Check if Tesseract is installed
    try:
        print("Tesseract Version:", pt.get_tesseract_version())
    except Exception as e:
        print("Error: Tesseract is not installed or not found in the specified path.")
        print(f"Details: {e}")
        return  # Exit the function instead of the script

    # Check if the image exists
    if not image_path.exists():
        print(f"Error: The image file '{image_path}' does not exist.")
        return

    try:
        # Load and process the image
        image = Image.open(image_path)

        # Extract text using OCR
        extracted_text = pt.image_to_string(image).strip()

        # Print the extracted text or notify if none is found
        if extracted_text:
            print("\nExtracted Text:\n", extracted_text)
        else:
            print("\nNo text detected in the image.")

    except Exception as e:
        print(f"Error: Failed to process the image. Details: {e}")


# Run the script when executed directly
if __name__ == "__main__":
    main()
