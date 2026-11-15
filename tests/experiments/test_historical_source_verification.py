import hashlib
import unittest
from unittest.mock import patch
from experiments.wind_bubble.compare_two_bubble_resolutions import verify_historical_sources


class HistoricalSources(unittest.TestCase):
    def test_exact_archived_bytes_required(self):
        revision='a'*40;old=b'archived source\n';digest=hashlib.sha256(old).hexdigest()
        with patch('experiments.wind_bubble.compare_two_bubble_resolutions.subprocess.check_output',return_value=old) as show:
            result=verify_historical_sources({'/export/home/mngo/master_thesis/example.py':digest},revision)
            self.assertEqual(result,{'example.py':digest})
            self.assertEqual(show.call_args.args[0],['git','show',revision+':example.py'])
        with patch('experiments.wind_bubble.compare_two_bubble_resolutions.subprocess.check_output',return_value=b'different bytes'):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_historical_sources({'example.py':digest},revision)

    def test_unrecorded_revision_and_nonrepo_paths_rejected(self):
        for revision in ['HEAD','abc',None]:
            with self.assertRaises(ValueError):verify_historical_sources({},revision)
        for name in ['../outside.py','/outside.py']:
            with self.assertRaises(ValueError):verify_historical_sources({name:'unused'},'a'*40)
