import torch

def contrastive_loss(embeddings, positive_pairs, negative_pairs, temperature):
    pos_sim = torch.cosine_similarity(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])
    neg_sim = torch.cosine_similarity(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
    loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))
    return loss.mean()