class SpecialSymbols(object):
    """
        Specials symbols that are used and added to target vocabuly
    """
    PAD = "_PAD"
    GO = "_GO"
    EOS = "_EOS"
    UNK = "_UNK"

    # pad is zero, because default value in the matrices is zero (np.zeroes)
    PAD_IX = 0
    GO_IX = 1
    EOS_IX = 2
    UNK_IX = 3
