import unittest

import torch

from evaluate_checkpoint_hits import NetReturnTracker


class NetMetricTest(unittest.TestCase):
    def setUp(self):
        self.t = NetReturnTracker(1, 'cpu')
        self.mask = torch.tensor([True])
        self.t.start(self.mask, self.mask, torch.tensor([[6., 0., 1.]]))

    def step(self, p, hit=False, ground=False):
        self.t.observe(torch.tensor([p]), torch.tensor([hit]), torch.tensor([ground]))

    def test_incoming_crossing_and_bounce_do_not_count(self):
        self.step([3., 0., 1.5])
        self.step([1., 0., .034], ground=True)
        self.assertFalse(self.t.success.item())
        self.assertFalse(self.t.landed.item())

    def test_late_contact_is_allowed_and_attempt_counts_once(self):
        self.step([1., 0., 1.], hit=True)
        self.step([3., 0., 1.4])
        self.step([4., 0., 1.2])
        self.t.finish(self.mask)
        self.t.finish(self.mask)
        self.assertEqual(self.t.counts['forehand'], dict(attempts=1, hits=1, contacts=1))

    def test_bounce_then_crossing_is_failure(self):
        self.step([1., 0., 1.], hit=True)
        self.step([2., 0., .034], ground=True)
        self.step([4., 0., 1.4])
        self.assertFalse(self.t.success.item())

    def test_low_or_outside_net_crossing_is_failure(self):
        for p in ([4., 0., .8], [4., 3., 1.5]):
            self.t.start(self.mask, self.mask, torch.tensor([[3., p[1], p[2]]]))
            self.step(p, hit=True)
            self.assertFalse(self.t.success.item())

    def test_crossing_without_racket_contact_is_failure(self):
        self.step([1., 0., 1.])
        self.step([4., 0., 1.5])
        self.assertFalse(self.t.success.item())

    def test_new_launch_clears_success_and_hand(self):
        self.step([1., 0., 1.], hit=True)
        self.step([4., 0., 1.5])
        self.t.finish(self.mask)
        self.t.start(self.mask, ~self.mask, torch.tensor([[6., 0., 1.]]))
        self.t.finish(self.mask)
        self.assertEqual(self.t.counts['forehand']['hits'], 1)
        self.assertEqual(self.t.counts['backhand'], dict(attempts=1, hits=0, contacts=0))


if __name__ == '__main__':
    unittest.main()
