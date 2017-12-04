class SpecialSymbols(object):
    # Special vocabulary symbols
    PAD = "_PAD"
    GO = "_GO"
    EOS = "_EOS"
    UNK = "_UNK"

    # pad is zero, because default value in the matrices is zero (np.zeroes)
    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3