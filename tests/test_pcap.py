import unittest
from pathlib import Path

from cicids_pipeline.pcap import find_pcap_files


class PcapDiscoveryTests(unittest.TestCase):
    def test_only_returns_real_pcap_files(self) -> None:
        root = Path(__file__).parent / "fixtures" / "pcaps"
        self.assertEqual(find_pcap_files(root), [root / "capture.pcap"])


if __name__ == "__main__":
    unittest.main()
