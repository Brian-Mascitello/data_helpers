import unittest
from unittest.mock import patch

import pandas as pd

import data_helpers.read_clipboard_io as rci


class TestReadClipboardIO(unittest.TestCase):

    def test_generate_filename_format(self):
        """Test that generate_filename produces a correctly formatted filename."""
        filename = rci.generate_filename()
        pattern = r"\d{8}_\d{6}_rci\.csv"
        self.assertRegex(filename, pattern)

    def test_save_dataframe_custom_path(self):
        """Test save_dataframe with a custom file path by patching DataFrame.to_csv."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with patch.object(df, "to_csv") as mock_to_csv:
            custom_path = "test_file.csv"
            rci.save_dataframe(df, custom_path)
            mock_to_csv.assert_called_once_with(
                custom_path, index=False, encoding="utf-8-sig"
            )

    def test_save_dataframe_auto_filename(self):
        """Test save_dataframe when no path is provided, ensuring auto-generated filename is used."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with patch.object(df, "to_csv") as mock_to_csv:
            with patch(
                "data_helpers.read_clipboard_io.generate_filename",
                return_value="auto_generated_rci.csv",
            ):
                rci.save_dataframe(df)
                mock_to_csv.assert_called_once_with(
                    "auto_generated_rci.csv", index=False, encoding="utf-8-sig"
                )

    def test_extract_and_copy_list_single_column(self):
        """Test extract_and_copy_list for a single column DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        with patch.object(pd.DataFrame, "to_clipboard") as mock_clipboard:
            # Test without clean_copy option enabled.
            result = rci.extract_and_copy_list(
                df, convert_to_str=True, clean_copy=False
            )
            self.assertEqual(result, [1, 2, 3])
            # Verify that to_clipboard was called.
            mock_clipboard.assert_called_once()

    def test_extract_and_copy_list_multi_column(self):
        """Test extract_and_copy_list for a multi-column DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with patch.object(pd.DataFrame, "to_clipboard") as mock_clipboard:
            result = rci.extract_and_copy_list(df, convert_to_str=False)
            # When convert_to_str is False, the values remain in their native types.
            self.assertEqual(result, [[1, 3], [2, 4]])
            # Check that to_clipboard is called with the formatted string.
            expected_str = "\n".join(
                " ".join(map(str, row)) for row in [[1, 3], [2, 4]]
            )
            mock_clipboard.assert_called_once()

    def test_read_from_clipboard_default(self):
        """Test read_from_clipboard using a default DataFrame from the clipboard."""
        sample_df = pd.DataFrame({"col": [1, 2, 3]})
        with patch(
            "data_helpers.read_clipboard_io.pd.read_clipboard", return_value=sample_df
        ) as mock_read:
            with patch.object(sample_df, "to_clipboard") as mock_to_clipboard:
                result = rci.read_from_clipboard(print_df=False, save_to_clipboard=True)
                mock_read.assert_called_once()
                mock_to_clipboard.assert_called_once()
                self.assertTrue(result.equals(sample_df))

    def test_read_from_clipboard_sort(self):
        """Test that read_from_clipboard sorts the DataFrame correctly."""
        sample_df = pd.DataFrame({"col": [3, 1, 2]})
        with patch(
            "data_helpers.read_clipboard_io.pd.read_clipboard", return_value=sample_df
        ):
            with patch.object(sample_df, "to_clipboard") as mock_to_clipboard:
                result = rci.read_from_clipboard(
                    print_df=False, sort=True, ascending=True
                )
                expected_df = sample_df.sort_values(by=["col"], ascending=True)
                pd.testing.assert_frame_equal(result, expected_df)

    def test_read_from_clipboard_return_list(self):
        """Test read_from_clipboard with return_list=True for single column DataFrame."""
        sample_df = pd.DataFrame({"col": [1, 2, 3]})
        with patch(
            "data_helpers.read_clipboard_io.pd.read_clipboard", return_value=sample_df
        ):
            with patch.object(pd.DataFrame, "to_clipboard"):
                result = rci.read_from_clipboard(return_list=True)
                # For a single column, extract_and_copy_list returns the column as a list.
                self.assertEqual(result, [1, 2, 3])

    def test_read_from_clipboard_save(self):
        """Test that read_from_clipboard calls save_dataframe when save=True."""
        sample_df = pd.DataFrame({"col": [1, 2, 3]})
        with patch(
            "data_helpers.read_clipboard_io.pd.read_clipboard", return_value=sample_df
        ):
            with patch(
                "data_helpers.read_clipboard_io.save_dataframe"
            ) as mock_save_dataframe:
                rci.read_from_clipboard(
                    return_list=True, save=True, save_path="test.csv"
                )
                mock_save_dataframe.assert_called_once_with(sample_df, "test.csv")

    def test_read_from_clipboard_exception(self):
        """Test read_from_clipboard's behavior when pd.read_clipboard raises an exception."""
        with patch(
            "data_helpers.read_clipboard_io.pd.read_clipboard",
            side_effect=Exception("Clipboard error"),
        ):
            result = rci.read_from_clipboard()
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
