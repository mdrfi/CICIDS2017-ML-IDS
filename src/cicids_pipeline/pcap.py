"""Helpers for PCAP files that may be added later."""

from pathlib import Path


PCAP_FOLDER = Path("dataset/PCAPs")


def find_pcap_files(folder: Path = PCAP_FOLDER) -> list[Path]:
    """Find real .pcap files and ignore CSV filenames containing '.pcap'."""
    if not folder.exists():
        return []
    return sorted(file for file in folder.rglob("*.pcap") if file.is_file())

