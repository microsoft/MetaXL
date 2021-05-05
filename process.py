import os
import glob

for f in glob.glob('*.tar.gz'):
    lan = f.split('.')[0]
    os.system('mkdir %s' % lan)
    os.system('tar -xvf %s -C %s' % (f, lan))
    print (f, lan)

