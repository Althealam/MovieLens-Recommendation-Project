import torch
from torch.utils.data import Dataset, DataLoader


class MovieDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        uid = torch.tensor(self.features[idx][0])
        movie_id = torch.tensor(self.features[idx][1])
        user_gender = torch.tensor(self.features[idx][2])
        user_age = torch.tensor(self.features[idx][3])
        user_job = torch.tensor(self.features[idx][4])
        movie_titles = torch.tensor(self.features[idx][6])
        movie_categories = torch.tensor(self.features[idx][7])
        history_movie_ids = torch.tensor(self.features[idx][8])

        targets = torch.tensor(self.targets[idx]).float()
        return uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, history_movie_ids, targets