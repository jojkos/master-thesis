from nmt import Translator, Vocabulary, SpecialSymbols
import numpy as np


def test_encode_sequences():
    x_word_seq = [
        ["jedna", "dva", "tři"],
        ["čtyři", "pět", "šest", "sedm", "osm"],
        ["devět"],
        ["deset"]
    ]
    x_max_seq_len = 5
    x_vocab = Vocabulary(x_word_seq, 100)
    
    y_word_seq = [
        ["one", "two", "three"],
        ["four", "five", "six", "seven"],
        ["eight"],
        ["nine", "ten"]
    ]
    y_max_seq_len = 4
    y_vocab = Vocabulary(y_word_seq, 100)

    reverse_input = False

    encoder_input_data, decoder_input_data, decoder_target_data = Translator.encode_sequences(
        x_word_seq=x_word_seq, y_word_seq=y_word_seq,
        x_max_seq_len=x_max_seq_len, y_max_seq_len=y_max_seq_len,
        source_vocab=x_vocab, target_vocab=y_vocab, reverse_input=reverse_input
    )

    test_encoder_input_data = np.asarray([
        [4, 5, 6, 0, 0],
        [7, 8, 9, 10, 11],
        [12, 0, 0, 0, 0],
        [13, 0, 0, 0, 0]
    ])
    np.testing.assert_array_equal(encoder_input_data, test_encoder_input_data)

    test_decoder_input_data = np.asarray([
        [4, 5, 6, 0],
        [7, 8, 9, 10],
        [11, 0, 0, 0],
        [12, 13, 0, 0]
    ])

    np.testing.assert_array_equal(decoder_input_data, test_decoder_input_data)

    test_decoder_target_data = np.asarray(
        [[[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
    )
    np.testing.assert_array_equal(decoder_target_data, test_decoder_target_data)


def test_get_gen_steps():
    class TestDataset(object):
        pass
    dataset = TestDataset()

    dataset.num_samples = 64
    batch_size = 64
    result = 1
    assert Translator.get_gen_steps(dataset, batch_size) == result

    dataset.num_samples = 63
    batch_size = 64
    result = 1
    assert Translator.get_gen_steps(dataset, batch_size) == result

    dataset.num_samples = 64
    batch_size = 63
    result = 2
    assert Translator.get_gen_steps(dataset, batch_size) == result

    dataset.num_samples = 127
    batch_size = 63
    result = 3
    assert Translator.get_gen_steps(dataset, batch_size) == result

    dataset.num_samples = 128
    batch_size = 64
    result = 2
    assert Translator.get_gen_steps(dataset, batch_size) == result


def test_encode_text_seq_to_encoder_seq():
    word_seq = [
        ["jedna", "dva", "tři"],
        ["čtyři", "pět", "šest", "sedm"],
        ["osm"],
        ["devět", "deset"]
    ]
    vocab = Vocabulary(word_seq, 100)

    text = "jedna dva kočka leze tři čtyři"

    test_encoded = np.asarray([[
        vocab.word_to_ix["jedna"], vocab.word_to_ix["dva"],
        SpecialSymbols.UNK_IX, SpecialSymbols.UNK_IX,
        vocab.word_to_ix["tři"], vocab.word_to_ix["čtyři"]
    ]], dtype="float32")

    encoded = Translator.encode_text_seq_to_encoder_seq(text, vocab)

    np.testing.assert_array_equal(encoded, test_encoded)
