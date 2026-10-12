"""Check spatial orientation and preservation of cluster assignments."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cluster_region_figures import nearest_sample_regions


class ClusterRegionTests(unittest.TestCase):
    def test_xy_orientation_and_sample_assignment_preservation(self):
        positions=np.array([[1,1],[6,1],[1,3],[6,3]])
        labels=np.array([0,1,2,3])
        grid,index,distance=nearest_sample_regions(positions,labels,(5,8))
        np.testing.assert_array_equal(grid[positions[:,1],positions[:,0]],labels)
        np.testing.assert_array_equal(grid,labels[index])
        self.assertEqual(grid[0,7],1)
        self.assertEqual(grid[4,0],2)
        self.assertTrue((distance>=0).all())

    def test_duplicate_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            nearest_sample_regions([[1,1],[1,1]],[0,1],(3,3))

    def test_mismatched_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            nearest_sample_regions([[1,1]],[0,1],(3,3))


if __name__=='__main__':
    unittest.main()
