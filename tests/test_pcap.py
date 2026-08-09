import unittest
from pathlib import Path

from cicids_pipeline.pcap import find_pcap_files


class PcapDiscoveryTests(unittest.TestCase):
    def test_missing_folder_returns_an_empty_list(self) -> None:
        self.assertEqual(find_pcap_files(Path("folder-that-does-not-exist")), [])


if __name__ == "__main__":
    unittest.main()
