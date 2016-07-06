import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import walk
import fnmatch

fileList = []
imagesFolder = './images'
for ( dirpath, dirnames, filenames ) in walk( imagesFolder ):
  fileList = fnmatch.filter(filenames, '*.png')
  break;


for file in fileList:
  print file
  img = cv2.imread(file, 0)

  # DFT
  dft = cv2.dft( np.float32( img ), flags = cv2.DFT_COMPLEX_OUTPUT )
  dft_shift = np.fft.fftshift( dft )
  magnitude_spectrum = 20 * np.log( cv2.magnitude( dft_shift[:,:,0], dft_shift[:,:,1] ) )

  #IDFT
  rows, cols = img.shape
  crow, ccol = rows/2, cols/2
  mask = np.ones((rows,cols,2), np.uint8)

  a = 30 
  mask[ crow - a : crow + a, ccol - a : ccol + a ] = 0

  fshift = dft_shift*mask
  f_ishift = np.fft.ifftshift( fshift )
  img_back = cv2.idft( f_ishift )
  img_back = cv2.magnitude( img_back[:, :, 0], img_back[:, :, 1] )

  img1 = 255 * ( img_back - np.amin(img_back) ) / ( np.amax( img_back ) - np.amin( img_back ) )
  cv2.imwrite( './fft/' + file + '.png', img1 )

  ## Artistic image
  org = cv2.imread( './fft/' + file + '.png', 0 )
  laplacian = cv2.Laplacian( org, cv2.CV_64F )
  cv2.imwrite( './laplacian/' + file + '.png', laplacian )
  laplacian = cv2.imread( './laplacian/' + file + '.png', 0 )
  blur = cv2.medianBlur( org, 21 )
  cv2.imwrite( './fftblur/' + file + '.png', laplacian - blur)




