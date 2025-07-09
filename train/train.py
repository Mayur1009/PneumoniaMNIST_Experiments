import numpy as np
from medmnist.dataset import PneumoniaMNIST
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


def binarize_images(imgs, ch=8):
    out = np.zeros((*imgs.shape, ch), dtype=np.uint32)
    for j in range(ch):
        t1 = (j + 1) * 255 / (ch + 1)
        out[:, :, :, j] = (imgs >= t1) & 1

    return out.reshape(imgs.shape[0], -1).astype(np.uint32)


def load_dataset(ch=8):
    train = PneumoniaMNIST(split="train", download=True)
    val = PneumoniaMNIST(split="val", download=True)
    test = PneumoniaMNIST(split="test", download=True)
    xtrain = binarize_images(train.imgs, ch)
    xval = binarize_images(val.imgs, ch)
    xtest = binarize_images(test.imgs, ch)
    return (
        (xtrain, train.labels.squeeze()),
        (xval, val.labels.squeeze()),
        (xtest, test.labels.squeeze()),
    )


def multiclass_metrics(true, pred, prob):
    acc = accuracy_score(true, pred)
    f1, precision, recall, _ = precision_recall_fscore_support(true, pred, average="macro")

    true_bin = np.zeros((len(true), np.max(true) + 1), dtype=np.uint32)
    true_bin[np.arange(len(true)), true] = 1
    roc = roc_auc_score(true_bin, prob, multi_class="ovr", average="macro")

    return {
        "accuracy": acc * 100,
        "f1": f1 * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "roc_auc": roc * 100,
    }


def print_metrics(epoch, train_met, val_met, test_met):
    col_width = 9
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    header = f"| {'Epoch = ' + str(epoch):^{col_width}} |"
    for metric in metrics:
        header += f" {metric:>{col_width}} |"
    print(header)
    separator = "+" + "+".join(["-" * (col_width+2)] * (len(metrics) + 1)) + "+"
    print(separator)
    for name, data in [("Train", train_met), ("Val", val_met), ("Test", test_met)]:
        row = f"| {name:>{col_width}} |"
        for metric in metrics:
            row += f" {data[metric]:>{col_width}.4f} |"
        print(row)
    print(separator)


def train(tm: MultiClassConvolutionalTsetlinMachine2D, xtrain, ytrain, xval, yval, xtest, ytest, epochs=1):
    for epoch in range(epochs):
        # Shuffle training data
        iota = np.arange(len(ytrain))
        np.random.shuffle(iota)
        xtrain = xtrain[iota]
        ytrain = ytrain[iota]

        # Fit the model
        tm.fit(xtrain, ytrain, epochs=1, incremental=True)

        # Evaluate Train
        cs_train = tm.score(xtrain)
        preds_train = np.argmax(cs_train, axis=1)

        # Convert class sums to probabilities
        prob_train = (np.clip(cs_train, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_train = prob_train / (np.sum(prob_train, axis=1, keepdims=True) + 1e-7)
        met_train = multiclass_metrics(ytrain, preds_train, prob_train)

        # Evaluate Validation
        cs_val = tm.score(xval)
        preds_val = np.argmax(cs_val, axis=1)
        prob_val = (np.clip(cs_val, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_val = prob_val / (np.sum(prob_val, axis=1, keepdims=True) + 1e-7)
        met_val = multiclass_metrics(yval, preds_val, prob_val)

        # Evaluate Test
        cs_test = tm.score(xtest)
        preds = np.argmax(cs_test, axis=1)
        prob_test = (np.clip(cs_test, -tm.T, tm.T) + tm.T) / (2 * tm.T)
        prob_test = prob_test / (np.sum(prob_test, axis=1, keepdims=True) + 1e-7)
        met_test = multiclass_metrics(ytest, preds, prob_test)

        # Print metrics
        print_metrics(epoch + 1, met_train, met_val, met_test)


if __name__ == "__main__":
    # Load dataset
    ch = 8
    (xtrain, ytrain), (xval, yval), (xtest, ytest) = load_dataset(ch=ch)

    # Initialize Tsetlin Machine
    tm = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses=100,
        T=500,
        s=5,
        dim=(28, 28, ch),
        patch_dim=(10, 10),
        q=1,
    )

    # Train the model
    train(tm, xtrain, ytrain, xval, yval, xtest, ytest, epochs=50)
