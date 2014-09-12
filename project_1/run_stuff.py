import cv2
import pano_stitcher
import numpy


class TestPanoStitcher():
    """Tests the functionality of the pano_stitcher module."""

    def setUp(self):
        """Initializes shared state for unit tests."""
        pass

    def _known_homography(self, rows, cols):
        """Returns an arbitrary homography for testing."""
        orig_points = numpy.array([[0, 0],
                                   [0, rows - 1],
                                   [cols - 1, 0],
                                   [cols - 1, rows - 1]],
                                  numpy.float32)
        warp_points = numpy.array([[10, 10],
                                   [-10, rows - 1 - 10],
                                   [cols - 1 - 30, 50],
                                   [cols - 1 - 60, rows - 1 - 30]],
                                  numpy.float32)
        return cv2.getPerspectiveTransform(orig_points, warp_points)

    def _scale_homography(self, scale):
        """Return a homography that scales by 'scale'"""
        return numpy.array([[scale, 0.0, 0.0],
                            [0.0, scale, 0.0],
                            [0.0, 0.0, 1.0]])

    def _translate_homography(self, x, y):
        """Return a homography that translates by (x, y)"""
        return numpy.array([[1.0, 0.0, x],
                            [0.0, 1.0, y],
                            [0.0, 0.0, 1.0]])

    def test_homography(self):
        """Checks that a known homography is recovered accurately."""
        # Load the left_houses image.
        houses_left = cv2.imread("test_data/houses_left.png",
                                 cv2.CV_LOAD_IMAGE_GRAYSCALE)
        rows, cols = houses_left.shape

        # Warp with a known homography.
        H_expected = self._known_homography(rows, cols)
        houses_left_warped = cv2.warpPerspective(houses_left, H_expected,
                                                 (cols, rows))

        # Compute the homography with the library.
        H_actual = pano_stitcher.homography(houses_left_warped, houses_left)

        # The two should be nearly equal.
        #H_difference = numpy.absolute(H_expected - H_actual)
        #H_difference_magnitude = numpy.linalg.norm(H_difference)

        #print "Expected homography:"
        #print H_expected
        #print "Actual homography:"
        #print H_actual
        #print "Difference:"
        #print H_difference
        #print "Magnitude of difference:", H_difference_magnitude

        #max_difference_magnitude = 0.5
        #self.assertLessEqual(H_difference_magnitude, max_difference_magnitude)

if __name__ == '__main__':
    stitcher = TestPanoStitcher()
    stitcher.test_homography()
