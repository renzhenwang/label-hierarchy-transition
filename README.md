
# label-hierarchy-transition
**Label Hierarchy Transition: Delving into Class Hierarchies to Enhance Deep Classifiers**

**Authors**: Renzhen Wang, De cai, Kaiwen Xiao, Xixi Jia, Xiao Han, Deyu Meng

[[`arXiv`](https://arxiv.org/abs/2112.02353)][[`BibTeX`](#Citation)]


**Introduction**: This is an official PyTorch implementation of [**"Label Hierarchy Transition: Delving into Class Hierarchies to Enhance Deep Classifiers"**](https://arxiv.org/abs/2112.02353). *In this paper, we propose Label Hierarchy Transition (LHT), a unified probabilistic framework based on deep learning, to address the challenges of hierarchical classification*. LHT is a simple, and efficient framework for hierarchical classification tasks, especially for missing label scenarios where a large number of samples are partially labeled at centain finer-level hierarchies.

## Requirements
* python==3.6
* numpy==1.19.2
* torch==1.10.2
* torchvision==0.11.3
* scikit-learn==0.24.2


## Dataset Preparation
* Download datasets
    * [CUB-200-201](https://www.vision.caltech.edu/datasets/cub_200_2011/)
    * [Aircraft](https://www.robots.ox.ac.uk/vgg/data/fgvc-aircraft)
    * [Stanford Car](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
* Extract the datasets to `data_source/cub_dataset/`, `data_source/air_dataset/` and `data_source/car_dataset/`, respectively.
* Reconstruct the datasets following the Birds.xls, Air.xls, and Cars.xls (borrowed from [Fine-Grained-or-Not](https://github.com/PRIS-CV/Fine-Grained-or-Not))
    ```
    cd DATASETNAME/datasets
    python data_precess.py
    ```
    `DATASETNAME` can be selected from `cub`,  `air`, and `car`.
* The prepocessed dataset is organized as follows.
    ```
    data_target
    ├── CUB_200_2011 
    │   ├── train 
    │   │   ├── 001.Black_footed_Albatross
    │   │   ├── 002.Laysan_Albatross
    │   │   ...
    │   └── test
    │       ├── 001.Black_footed_Albatross  
    │       ...            
    ├── Aircraft
    │   ├── train
    │   ... 
    └── Cars
        ├── train
        ...    
    ```




## Setup and Training

* Training our method **LHT** (e.g, **CUB-200-2011**):

  * For example, run the experiment with relabeled ratio **$\mathcal R=90\%$**.
    ```
    cd cub
    python main_ours.py ../data_target/CUB-200-2011 -b 8 -j 4 --lr 0.002 --epochs 200 --crop_size 448 --scale_size 550 --ratio 0.1 --beta 0.01 --seed 10 --out ./result/ours@ratio_0.1_seed_10 
    ```

  * Or, one can directly run the experiment with relabeled ratio **$\mathcal R=90\%$** three times (random seeds 10, 100, 1) as follows.
    ```
    cd cub
    run run_ours.sh
    ```

* The saved folder (including logs and checkpoints) is organized as follows.
    ```
    cub
    ├── result 
    │   ├── ours@ratio_0.1_seed_10
    │   │   ├── checkpoint.pth.tar
    │   │   ├── log_train-*.txt
    │   │   └── model_best.pth.tar
    │   ...
    ...
    ```



## <a name="Citation"></a>Citation

If you find our work or this code is useful, please cite us:

```
@rticle{LHT,
  title={Label Hierarchy Transition: Delving into Class Hierarchies to Enhance Deep Classifiers},
  author={Renzhen Wang and De Cai and Kaiwen Xiao and Xixi Jia and Xiao Han and Deyu Meng},
  journal={CoRR},
  volume={abs/2112.02353},
  year={2021},
  url={https://arxiv.org/abs/2112.02353}
}
```




## Questions
Please feel free to contact Renzhen Wang ([rzwang@xjtu.edu.cn](mailto:rzwang@xjtu.edu.cn)) or Deyu Meng ([dymeng@xjtu.edu.cn](mailto:dymeng@xjtu.edu.cn)).
