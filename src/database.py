import os
import cv2
import numpy as np
from tqdm import tqdm

class IdentityDatabase:
    def __init__(self):
        self.embeddings = {}
        self.reference_face = {}
    def add(self, name, embedding):
        self.embeddings[name] = embedding

    def load_from_folder(self, folder, detector, embedder):
        """
        folder/
          person1/*.jpg
          person2/*.jpg
        """
        for person in os.listdir(folder):
            person_dir = os.path.join(folder, person)
            if not os.path.isdir(person_dir):
                continue

            person_embeddings = []

            for img_name in tqdm(os.listdir(person_dir), desc=f"Loading {person}"):
                img = cv2.imread(os.path.join(person_dir, img_name))
                faces = detector.detect(img)
                if len(faces) == 0:
                    continue
                x1, y1, x2, y2 = faces[0].bbox.astype(int)
                emb = embedder.extract([faces[0]])[0]
                reference_face = img[y1:y2, x1:x2].copy()
                person_embeddings.append(emb)

            if person_embeddings:
                self.reference_face[person] = reference_face
                self.embeddings[person] = np.mean(person_embeddings, axis=0)

        return self.reference_face, self.embeddings