import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class NewsgroupsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx], "label": self.targets[idx]}

# Load the dataset
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Preprocess the text data: Here, we use TF-IDF vectorization. You can also use word embeddings.
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
X_test = vectorizer.transform(newsgroups_test.data).toarray()

y_train, y_test = newsgroups_train.target, newsgroups_test.target

# Creating the custom dataset
train_dataset = NewsgroupsDataset(X_train, y_train)
test_dataset = NewsgroupsDataset(X_test, y_test)

# Creating the DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print("Train dataset size: ", len(train_dataset))
    print("Test dataset size: ", len(test_dataset))
    print("Train dataloader size: ", len(train_loader))
    print("Test dataloader size: ", len(test_loader))

    for batch in train_loader:
        print(batch["text"].shape)
        print(batch["label"].shape)
        break

    for batch in test_loader:
        print(batch["text"].shape)
        print(batch["label"].shape)
        break

    