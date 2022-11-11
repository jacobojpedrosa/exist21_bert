from datasets import load_dataset


def text_to_binary(examples):
    if examples["task1"] == 'non-sexist':
        binary_vec = 0
    else:
        binary_vec = 1

    return {"label": binary_vec}


def data_preparation(path_train, path_test):
    dataset = load_dataset("csv", data_files={"train": path_train,
                                              "test": path_test}, sep='\t')

    dataset = dataset.map(text_to_binary)
    en_dataset = dataset.filter(lambda example: example["language"] == 'en')
    es_dataset = dataset.filter(lambda example: example["language"] == 'es')

    return en_dataset, es_dataset
