from nmt import Translator, Vocabulary, SpecialSymbols
import numpy as np
import os
import random
import shutil

# TODO so that tests run the same way every time
random.seed(0)


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """

    shutil.rmtree('logs')


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
        [SpecialSymbols.GO, "one", "two", "three", SpecialSymbols.EOS],
        [SpecialSymbols.GO, "four", "five", "six", "seven", SpecialSymbols.EOS],
        [SpecialSymbols.GO, "eight", SpecialSymbols.EOS],
        [SpecialSymbols.GO, "nine", "ten", SpecialSymbols.EOS]
    ]
    y_max_seq_len = 6
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
        [SpecialSymbols.GO_IX, 4, 5, 6, 0],
        [SpecialSymbols.GO_IX, 7, 8, 9, 10],
        [SpecialSymbols.GO_IX, 11, 0, 0, 0],
        [SpecialSymbols.GO_IX, 12, 13, 0, 0]
    ])

    np.testing.assert_array_equal(decoder_input_data, test_decoder_input_data)

    decoded_target_data = []
    for seq in decoder_target_data:
        decoded_target_data.append(
            Translator.decode_encoded_seq(seq, y_vocab, one_hot=True)
        )

    test_target_data = [
        ["one", "two", "three", SpecialSymbols.EOS, SpecialSymbols.PAD],
        ["four", "five", "six", "seven", SpecialSymbols.EOS],
        ["eight", SpecialSymbols.EOS, SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD],
        ["nine", "ten", SpecialSymbols.EOS, SpecialSymbols.PAD, SpecialSymbols.PAD]
    ]

    np.testing.assert_array_equal(decoded_target_data, test_target_data)


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


def test_translating_small_dataset():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")

    translator.fit(epochs=100)
    translator.evaluate(beam_size=5)

    os.remove("data/model.h5")

    with open("data/smallTest.en.reference.translated", encoding="utf-8") as test_file:
        test_data = test_file.read()
    with open("data/smallTest.en.translated", encoding="utf-8") as translated_file:
        translated_data = translated_file.read()

    os.remove("data/smallTest.en.translated")

    assert translated_data == test_data


def test_translating_small_dataset_use_generator():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")

    translator.fit(epochs=100, use_fit_generator=True)
    translator.evaluate()

    os.remove("data/model.h5")

    with open("data/smallTest.en.reference.translated", encoding="utf-8") as test_file:
        test_data = test_file.read()
    with open("data/smallTest.en.translated", encoding="utf-8") as translated_file:
        translated_data = translated_file.read()

    os.remove("data/smallTest.en.translated")

    assert translated_data == test_data


def test_translating_small_dataset_bucketing():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")

    translator.fit(epochs=100, bucketing=True, bucket_range=2)

    translator.evaluate()

    os.remove("data/model.h5")

    with open("data/smallTest.en.reference.translated", encoding="utf-8") as test_file:
        test_data = test_file.read()
    with open("data/smallTest.en.translated", encoding="utf-8") as translated_file:
        translated_data = translated_file.read()

    os.remove("data/smallTest.en.translated")

    assert translated_data == test_data


def test_translating_small_dataset_multiple_layers():
    # TODO multiple decoder layers not working properly
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5",
                            num_encoder_layers=4, num_decoder_layers=2)

    translator.fit(epochs=100, bucketing=True, bucket_range=2)

    translator.evaluate()

    os.remove("data/model.h5")

    with open("data/smallTest.en.reference.translated", encoding="utf-8") as test_file:
        test_data = test_file.read()
    with open("data/smallTest.en.translated", encoding="utf-8") as translated_file:
        translated_data = translated_file.read()

    os.remove("data/smallTest.en.translated")

    assert translated_data == test_data


def test_get_training_data():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")

    training_data = translator._get_training_data()

    decoded_data = Translator.decode_encoded_seq(training_data["encoder_input_data"][0], translator.source_vocab)
    test_decoded_data = ["se", "rozzlobila", SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(training_data["decoder_input_data"][0], translator.target_vocab)
    test_decoded_data = [SpecialSymbols.GO, "she", "got", "angry", SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(training_data["decoder_target_data"][0], translator.target_vocab,
                                                 one_hot=True)
    test_decoded_data = ["she", "got", "angry", SpecialSymbols.EOS, SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)


def test_training_data_gen():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")

    generator = translator._training_data_gen(batch_size=4, shuffle=False)

    # to remove first returned value
    steps = next(generator)

    training_data = next(generator)
    encoder_input_data = training_data[0][0]
    decoder_input_data = training_data[0][1]
    decoder_target_data = training_data[1]

    assert len(encoder_input_data) == 4
    assert len(decoder_input_data) == 4
    assert len(decoder_target_data) == 4

    decoded_data = Translator.decode_encoded_seq(encoder_input_data[0], translator.source_vocab)
    test_decoded_data = ["se", "rozzlobila", SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(decoder_input_data[0], translator.target_vocab)
    test_decoded_data = [SpecialSymbols.GO, "she", "got", "angry", SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(decoder_target_data[0], translator.target_vocab,
                                                 one_hot=True)
    test_decoded_data = ["she", "got", "angry", SpecialSymbols.EOS, SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    training_data = next(generator)
    encoder_input_data = training_data[0][0]
    decoder_input_data = training_data[0][1]
    decoder_target_data = training_data[1]

    assert len(encoder_input_data) == 3
    assert len(decoder_input_data) == 3
    assert len(decoder_target_data) == 3


def test_training_data_gen_shuffling():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")
    random.seed(1)  # seed chosen to switch the indeces in data generator
    generator = translator._training_data_gen(batch_size=4, shuffle=True)

    # to remove first returned value
    steps = next(generator)

    training_data = next(generator)
    encoder_input_data = training_data[0][0]
    decoder_input_data = training_data[0][1]
    decoder_target_data = training_data[1]

    assert len(encoder_input_data) == 4
    assert len(decoder_input_data) == 4
    assert len(decoder_target_data) == 4

    training_data = next(generator)
    encoder_input_data = training_data[0][0]
    decoder_input_data = training_data[0][1]
    decoder_target_data = training_data[1]

    assert len(encoder_input_data) == 3
    assert len(decoder_input_data) == 3
    assert len(decoder_target_data) == 3

    decoded_data = Translator.decode_encoded_seq(encoder_input_data[0], translator.source_vocab)
    test_decoded_data = ["se", "rozzlobila", SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(decoder_input_data[0], translator.target_vocab)
    test_decoded_data = [SpecialSymbols.GO, "she", "got", "angry", SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(decoder_target_data[0], translator.target_vocab,
                                                 one_hot=True)
    test_decoded_data = ["she", "got", "angry", SpecialSymbols.EOS, SpecialSymbols.PAD, SpecialSymbols.PAD,
                         SpecialSymbols.PAD, SpecialSymbols.PAD, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)


def test_training_data_gen_bucketing():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")
    generator = translator._training_data_bucketing(batch_size=2, infinite=True,
                                                    shuffle=False, bucketing=True,
                                                    bucket_range=1)

    # to remove first returned value
    steps = next(generator)
    assert steps == 4

    training_data = next(generator)
    encoder_input_data = training_data[0][0]
    decoder_input_data = training_data[0][1]
    decoder_target_data = training_data[1]

    assert len(encoder_input_data) == 2
    assert len(decoder_input_data) == 2
    assert len(decoder_target_data) == 2

    training_data = next(generator)
    encoder_input_data = training_data[0][0]
    decoder_input_data = training_data[0][1]
    decoder_target_data = training_data[1]

    assert len(encoder_input_data) == 1
    assert len(decoder_input_data) == 1
    assert len(decoder_target_data) == 1

    decoded_data = Translator.decode_encoded_seq(encoder_input_data[0], translator.source_vocab)
    test_decoded_data = ["přátelé", "jsme", SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(decoder_input_data[0], translator.target_vocab)
    test_decoded_data = [SpecialSymbols.GO, "we're", "friends", SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)

    decoded_data = Translator.decode_encoded_seq(decoder_target_data[0], translator.target_vocab,
                                                 one_hot=True)
    test_decoded_data = ["we're", "friends", SpecialSymbols.EOS, SpecialSymbols.PAD]
    np.testing.assert_array_equal(decoded_data, test_decoded_data)


def test_define_models_default():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5")

    model, encoder_model, decoder_model = translator._define_models()

    model_layers = ["encoder_input", "input_embeddings", "decoder_input", "bidirectional_encoder_layer",
                    "target_embeddings", "average_3", "average_4", "decoder_layer_1", "output_layer"]

    encoder_input = model.get_layer(name="encoder_input")
    input_embeddings = model.get_layer(name="input_embeddings")
    bidirectional_encoder_layer = model.get_layer(name="bidirectional_encoder_layer")
    decoder_input = model.get_layer(name="decoder_input")
    target_embeddings = model.get_layer(name="target_embeddings")
    decoder_layer_1 = model.get_layer(name="decoder_layer_1")
    output_layer = model.get_layer(name="output_layer")

    assert len(model.layers) == 9
    assert encoder_input.get_output_at(0) == input_embeddings.get_input_at(0)
    assert input_embeddings.get_output_at(0) == bidirectional_encoder_layer.get_input_at(0)
    assert decoder_input.get_output_at(0) == target_embeddings.get_input_at(0)
    assert decoder_layer_1.get_output_at(0)[0] == output_layer.get_input_at(0)

    assert len(decoder_model.layers) == 6
    assert decoder_model.get_layer("decoder_layer_1").get_input_at(0)[0] == decoder_model.get_layer(
        "target_embeddings").get_output_at(0)

    assert len(encoder_model.layers) == 5


def test_define_models_multiple_layers():
    translator = Translator(training_dataset="data/small", test_dataset="data/smallTest",
                            source_lang="cs", target_lang="en", log_folder="logs",
                            model_folder="data", model_file="model.h5",
                            num_encoder_layers=2, num_decoder_layers=2)

    model, encoder_model, decoder_model = translator._define_models()

    encoder_input = model.get_layer(name="encoder_input")
    input_embeddings = model.get_layer(name="input_embeddings")
    bidirectional_encoder_layer = model.get_layer(name="bidirectional_encoder_layer")
    decoder_input = model.get_layer(name="decoder_input")
    target_embeddings = model.get_layer(name="target_embeddings")
    decoder_layer_1 = model.get_layer(name="decoder_layer_1")
    output_layer = model.get_layer(name="output_layer")

    assert len(model.layers) == 11

    assert len(decoder_model.layers) == 10

    assert len(encoder_model.layers) == 5

    # TODO testy pro vice layeru encoder/decoder a mozna bych mel udelat test co overi ze vubec spravne vytvarim modely a maji spravnej pocet vrstev a tak
