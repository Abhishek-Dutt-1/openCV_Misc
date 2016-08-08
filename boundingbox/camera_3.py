import numpy as np
import cv2
import subprocess as sp

from goprohero import GoProHero

camera = GoProHero(password='9060564940')
#camera.command('record', 'on')
#status = camera.status()

#camera.command('power', 'on')
#camera.command('mode', 'still')
camera.command('mode', 'video')
#camera.command('picres', '5MP wide')
#camera.command('vidres', '720p')
#camera.command('vidres', '960p')
camera.command('vidres', '1080p')
#camera.command('fov', '90')
#status = camera.status()
#password = camera.password()
image = camera.image()
