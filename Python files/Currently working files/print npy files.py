import numpy as np

print('zu:',np.load('moids1000zup.npy').tolist())
print('ul:',np.load('moids1000upleft.npy').tolist())
print('ur:',np.load('moids1000upright.npy').tolist())
print('dl:',np.load('moids1000downleft.npy').tolist())
print('up:',np.load('moids1000up.npy').tolist())
print('do:',np.load('moids1000down.npy').tolist())
print('dr:',np.load('moids1000downright.npy').tolist())