import numpy as np
import matplotlib.pyplot as plt
import cv2

test_img = np.load("Arrays/D3_Test.npy")

test_img = np.squeeze(test_img, axis=0)
test_img = np.squeeze(test_img, axis=2)
# plt.figure()
# plt.grid(False)
# plt.imshow(test_img, cmap='gray')
# plt.show()

cv2.imshow('test_image', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
