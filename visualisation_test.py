import numpy as np
import torch

if __name__=='__main__':

    with np.load("/home/nnair/DFIdemo/test.npz") as data:
        real=data['real']
        fake=data['fake']
    
    print(f'real shape: {real.shape}')
    print(f'fake.shape: {fake.shape}')
    print(f'real type: {real.dtype}')
    print(f'fake.type: {fake.dtype}')
    print(f'real:{real}')
    print(f'real:{real.shape}')
    squeezed=np.squeeze(real, axis=1)
    print(squeezed)
    print(squeezed.shape)
    np.save('/home/nnair/DFIdemo/real.npy', squeezed)
    print('real saved')
    print(f'fake:{fake}')
    print(f'fake:{fake.shape}')
    squeezed=np.squeeze(fake, axis=1)
    print(squeezed)
    print(squeezed.shape)
    np.save('/home/nnair/DFIdemo/fake.npy', squeezed)
    print('fake saved')
