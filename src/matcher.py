import numpy as np

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


class IdentityMatcher:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def match(self, embedding, database):
        """
        database: dict[name -> embedding]
        """
        best_name = "unknown"
        best_score = -1.0

        for name, db_emb in database.items():
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < self.threshold:
            return "unknown", best_score

        return best_name, best_score