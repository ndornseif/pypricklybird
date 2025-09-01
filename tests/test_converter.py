"""Test the functionality of the conversion between pricklybird strings and bytes."""

# ruff: noqa: INP001

import pytest

from pypricklybird import (
    CRC8_TABLE,
    CRCError,
    DecodeError,
    bytes_to_words,
    calculate_crc8,
    convert_from_pricklybird,
    convert_to_pricklybird,
    words_to_bytes,
)


@pytest.fixture(name="test_data")
def _fixture_test_data() -> bytearray:
    """Generate pseudorandom test data using the Lehmer64 LCG."""
    seed = 1
    output_bytes = 4096
    warmup_iterations = 128

    state = seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    multiplier = 0xDA942042E4DD58B5
    # Mix up the state a little to compensate for potentialy small seed.
    for _ in range(warmup_iterations):
        state = (state * multiplier) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

    result = bytearray(output_bytes)
    for i in range(0, output_bytes, 8):
        state = (state * multiplier) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        random_val = (state >> 64) & 0xFFFFFFFFFFFFFFFF
        result[i : i + 8] = random_val.to_bytes(8, "little")
    return result


class TestConversion:
    """Test the conversion between pricklybird strings and bytes."""

    @pytest.mark.parametrize(
        ("data", "words"),
        [
            (bytearray.fromhex("DEADBEEF"), "turf-port-rust-warn-void"),
            (bytearray.fromhex("4243"), "flea-flux-full"),
            (bytearray.fromhex("1234567890"), "blob-eggs-hair-king-meta-yell"),
            (bytearray.fromhex("0000000000"), "acid-acid-acid-acid-acid-acid"),
            (bytearray.fromhex("ffffffffff"), "zone-zone-zone-zone-zone-sand"),
        ],
    )
    def test_vectors(self, data: bytearray, words: str) -> None:
        """Test the standard vectors supplied with the specification."""
        assert words == convert_to_pricklybird(data), (
            f"Failed to convert {data} to pricklybird"
        )
        assert data == convert_from_pricklybird(words), (
            f"Failed to convert {words} to bytes"
        )

    def test_simple_conversion(self, test_data: bytearray) -> None:
        """Test conversion to and from pricklybird on pseudorandom test data."""
        coded_words = convert_to_pricklybird(test_data)
        assert test_data == convert_from_pricklybird(coded_words), (
            "Converter did not correctly encode or decode data"
        )

    def test_uppercase(self) -> None:
        """Test that pricklybird input containing mixed case is properly decoded."""
        result = convert_from_pricklybird("TUrF-Port-RUST-warn-vOid")
        expected = bytearray.fromhex("DEADBEEF")
        assert result == expected, "Converter did not correctly decode uppercase data"

    def test_error_detection_bit_flip(self, test_data: bytearray) -> None:
        """Test that replacing a pricklybird word is detected using the CRC-8."""
        coded_words = convert_to_pricklybird(test_data)
        # Flip bit in first word
        test_data[0] ^= 1
        incorrect_word = bytes_to_words(test_data[0:1])[0]

        # Replace fist word with incorrect one
        corrupted_words = incorrect_word[:4] + coded_words[4:]
        with pytest.raises(CRCError):  # noqa: PT012
            convert_from_pricklybird(corrupted_words)
            pytest.fail("Converter did not detect error in corrupted input")

    def test_error_detection_adjacent_swap(self, test_data: bytearray) -> None:
        """Check that swapping two adjacent words is detected using the CRC-8."""
        coded_words = convert_to_pricklybird(test_data)
        word_list = coded_words.split("-")
        word_list[0], word_list[1] = word_list[1], word_list[0]
        swapped_coded_words = "-".join(word_list)
        with pytest.raises(CRCError):  # noqa: PT012
            convert_from_pricklybird(swapped_coded_words)
            pytest.fail("Converter did not detect error caused by word swap.")

    def test_whitespace_trim(self) -> None:
        """Check that whitespace is correctly trimmed."""
        assert bytearray.fromhex("4243") == convert_from_pricklybird(
            " \t\n\r\x0b\x0c flea-flux-full \t\n\r\x0b\x0c "
        )

    @pytest.mark.parametrize(
        ("input_words", "msg"),
        [
            ("", "Converter did not error with empty input"),
            ("orca", "Converter did not error with too short input"),
            ("a®¿a-orca", "Converter did not error with non ASCII input"),
            (
                "gäsp-risk-king-orca-husk",
                "Converter did not error with non ASCII input",
            ),
            (
                "-risk-king-orca-husk",
                "Converter did not error with incorrectly formatted input",
            ),
            (
                "gasp-rock-king-orca-husk",
                "Converter did not error with incorrect word in input",
            ),
            ("flea- \t \t-full", "Converter did not error with whitespace in input"),
            ("flea-aaa\0-full", "Converter did not error with null bytes in input"),
            ("flea-\0aaa-full", "Converter did not error with null bytes in input"),
            (
                "flea-\x7faaa-full",
                "Converter did not error with ASCII control character in input",
            ),
            (
                "flea-aaa\x7f-full",
                "Converter did not error with ASCII control character in input",
            ),
            ("zzzz-king", "Converter did not error with incorrect word in input"),
        ],
    )
    def test_unusual_input(self, input_words: str, msg: str) -> None:
        """Check that edge cases result in the correct errors."""
        with pytest.raises(DecodeError):  # noqa: PT012
            convert_from_pricklybird(input_words)
            pytest.fail(msg)

    def test_empty_input(self) -> None:
        """Check that empty input results in empty output."""
        assert convert_to_pricklybird(bytearray()) == ""
        assert bytes_to_words(bytearray()) == []
        assert bytearray() == words_to_bytes([])


class TestCRC8:
    """Check functionality of the cyclic redundancy check."""

    def test_empty_input(self) -> None:
        """Check that CRC-8 of empty input is zero."""
        assert calculate_crc8(bytearray()) == b"\x00", (
            "CRC-8 of empty data should be 0."
        )

    def test_table_lookup(self) -> None:
        """Check that CRC-8 of a byte is equal to the matching table value."""
        test_data = bytearray([0x42])
        result = calculate_crc8(test_data)[0]
        expected = CRC8_TABLE[test_data[0]]
        assert expected == result, "CRC-8 of single byte should match table value."

    def test_with_appended_crc(self) -> None:
        """Check that data with appended correct CRC-8 has a remainder of zero."""
        test_data = bytearray(b"Test data")
        test_data.extend(calculate_crc8(test_data))
        assert calculate_crc8(test_data) == b"\x00", (
            "Data with appended correct CRC-8 should result in remainder 0."
        )
