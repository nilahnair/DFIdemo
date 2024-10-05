# This code is attributed to Dr.-Ing. Fernando Moya Rueda <https://github.com/wilfer9008>
#
def size_feature_map(Wx, Hx, F, P, S, type_layer = 'conv'):
    '''
    Computing size of feature map after convolution or pooling

    @param Wx: Width input
    @param Hx: Height input
    @param F: Filter size
    @param P: Padding
    @param S: Stride
    @param type_layer: conv or pool
    @return Wy: Width output
    @return Hy: Height output
    '''

    Pw = P
    Ph = P

    if type_layer == 'conv':
        Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
        Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

    elif type_layer == 'pool':
        Wy = 1 + (Wx - F[0]) / S[0]
        Hy = 1 + (Hx - F[1]) / S[1]

    return Wy, Hy