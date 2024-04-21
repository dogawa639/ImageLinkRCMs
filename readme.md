# Route Choice Models with Link Image

## Abstract
Main components
- Preprocessing
- Model
- AIRL route choice model
- Recursive logit model
- Logger

## Requirements
See `requirements.txt`
## Usage
1. Create a new conda environment
```bash
conda create -n new_env --file requirements.txt -c conda-forge -c pytorch
```
2. Activate the environment
```bash
conda activate new_env
```
3. Add the project folder to the python path in your IDE settings
4. Change the paths in the config files to the correct paths on your machine
5. Run the main python files in the `main` folder

## Structure
```mermaid
graph LR
    %% GPS
    A --> F;
    D --> L;
    F --> L;
    L --> U;
    
    A --> H;
    E --> N;
    H --> N;
    O --> X;
    
    C --> P;
    P --> Q;
    Q --> L;
    R --> N;
    
    %% Camera
    B --> G;
    G --> H;
    E --> M;
    H -->M;
    M --> V;
    
    C --> R;
    R --> M;
    
    %% multi scale
    L --> Y;
    M --> Y;
    
    %% Logger
    AA --> Z;
    
    %% sub module
    D <--> K;
    N --> O;
    I <--> J;
    S --> T;
    T -.->U;
    T -.-> V;
    T -.-> X;
    T -.-> Y;
    
    subgraph DataBehavior
        A[(GPS)];
        B[(Camera)];
    end
    subgraph Feature
        C[Image];
    end
    subgraph preprocessing
        D[network];
        E[mesh];
        F[pp];
        G[movie];
        H[mesh_trajectory];
        J[geo_util];
        K[osm_util];
        subgraph I[image]
            P[LinkImageData];
            Q[CompressedImageData];
            R[MeshImageData];
        end
        subgraph dataset
            direction TB
            L[GridDataset<br>PPEmbedDataset];
            M[MeshDataset];
            N[MeshDatasetStatic];
            O[MeshDatasetStaticSub];
        end
    end
    S[models];
    subgraph AA[learning]
        T[util];
        U[airl];
        V[mesh_airl];
        X[mesh_airl_static];
        Y[multi_scale_airl];
    end
    Z[Logger];
```

## References
- Oyama, Y., & Hato, E. (2017). A discounted recursive logit model for dynamic gridlock network analysis. Transportation Research Part C: Emerging Technologies, 85, 509–527.
- Zhao, Z., & Liang, Y. (2023). A deep inverse reinforcement learning approach to route choice modeling with context-dependent rewards. Transportation Research Part C: Emerging Technologies, 149, 104079.
- Alsaleh, R., & Sayed, T. (2021). Markov-game modeling of cyclist-pedestrian interactions in shared spaces: A multi-agent adversarial inverse reinforcement learning approach. Transportation Research Part C: Emerging Technologies, 128, 103191. https://doi.org/10.1016/j.trc.2021.103191
## License
MIT