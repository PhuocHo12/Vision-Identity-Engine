# Vision-Identity-Engine
Core identity recognition engine for real-time vision systems

## Architecture 
```mermaid
flowchart TD
    A[Input Frame]
    B["Face Detection 
    (RetinaFace)"]
    C["Embedding Extraction 
    (ArcFace)"]
    D[Similarity Matching]
    E[Identity Decision]

    A --> B
    B --> C
    C --> D
    D --> E
  ```