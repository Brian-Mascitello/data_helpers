"""
NOTE: You must install Tesseract OCR on Windows before using this script.
Download from: https://github.com/UB-Mannheim/tesseract/wiki
After installation, ensure Tesseract is added to the system PATH or specify the full path below.
"""

from datetime import datetime
from pathlib import Path

import pytesseract as pt
from PIL import Image


def check_tesseract_installed(tesseract_path: Path, verbose: bool = True) -> bool:
    """
    Checks if Tesseract OCR is installed and accessible.

    Args:
        tesseract_path (Path): The path to the Tesseract executable.
        verbose (bool): Whether to print messages to the terminal.

    Returns:
        bool: True if Tesseract is installed and the version can be retrieved, False otherwise.
    """
    pt.pytesseract.tesseract_cmd = str(tesseract_path)
    try:
        version = pt.get_tesseract_version()
        if verbose:
            print("Tesseract Version:", version)
        return True
    except Exception as e:
        if verbose:
            print(
                "Error: Tesseract is not installed or not found in the specified path."
            )
            print(f"Details: {e}")
        return False


def process_image(image_path: Path, verbose: bool = True) -> str:
    """
    Loads an image and extracts text using Tesseract OCR.

    Args:
        image_path (Path): The path to the image file.
        verbose (bool): Whether to print messages to the terminal.

    Returns:
        str: The extracted text. Returns an empty string if the image does not exist or processing fails.
    """
    if not image_path.exists():
        if verbose:
            print(f"Error: The image file '{image_path}' does not exist.")
        return ""
    try:
        image = Image.open(image_path)
        extracted_text = pt.image_to_string(image).strip()
        return extracted_text
    except Exception as e:
        if verbose:
            print(f"Error: Failed to process the image. Details: {e}")
        return ""


def save_text_to_file(
    text: str,
    base_filename: str = "extracted_text.txt",
    datetime_stamp: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Saves the provided text to a file. Optionally includes the current datetime in the filename.

    Args:
        text (str): The text to save.
        base_filename (str): The base filename for the output file.
        datetime_stamp (bool): If True, appends a datetime stamp to the filename.
        verbose (bool): Whether to print messages to the terminal.

    Returns:
        Path: The path to the saved text file.
    """
    if datetime_stamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = base_filename.rsplit(".", 1)
        if len(filename_parts) == 2:
            filename = f"{filename_parts[0]}_{timestamp}.{filename_parts[1]}"
        else:
            filename = f"{base_filename}_{timestamp}"
    else:
        filename = base_filename

    file_path = Path(filename)
    try:
        file_path.write_text(text, encoding="utf-8")
        if verbose:
            print(f"Text saved to file: {file_path}")
    except Exception as e:
        if verbose:
            print(f"Error: Failed to save text to file. Details: {e}")
    return file_path


def perform_ocr(
    tesseract_path: Path,
    file_to_process: Path,
    output_filename: str,
    use_datetime_stamp: bool,
    verbose: bool = True,
) -> None:
    """
    Checks if Tesseract is installed and processes an image file using OCR.
    Extracts text and saves it to a file if text is found.

    Args:
        tesseract_path (Path): The path to the Tesseract executable.
        file_to_process (Path): The image file to process.
        output_filename (str): The base filename for the output text file.
        use_datetime_stamp (bool): If True, a datetime stamp is added to the output filename.
        verbose (bool): Whether to print messages to the terminal.
    """
    # Check if Tesseract is installed and accessible
    if not check_tesseract_installed(tesseract_path, verbose):
        return

    # Process the image file
    if verbose:
        print("Processing image file...")
    extracted_text = process_image(file_to_process, verbose)

    # Display and save the extracted text if any is found
    if extracted_text:
        if verbose:
            print("\nExtracted Text:\n", extracted_text)
        save_text_to_file(
            extracted_text,
            base_filename=output_filename,
            datetime_stamp=use_datetime_stamp,
            verbose=verbose,
        )
    else:
        if verbose:
            print("\nNo text detected in the file.")


def main():
    """
    Main function to define settings and run the OCR process.
    """
    # User-defined settings (easily editable)
    tesseract_path = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )  # Set Tesseract path
    file_to_process = Path("test_image.png")  # Change to your image file path
    output_filename = "extracted_text.txt"  # Base filename for output
    use_datetime_stamp = True  # Include a datetime stamp in the output filename if True
    verbose = True  # Set to False to suppress terminal output

    # Run OCR processing on the provided file
    perform_ocr(
        tesseract_path=tesseract_path,
        file_to_process=file_to_process,
        output_filename=output_filename,
        use_datetime_stamp=use_datetime_stamp,
        verbose=verbose,
    )


# Run the script when executed directly
if __name__ == "__main__":
    main()
