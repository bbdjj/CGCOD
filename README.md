# CGCOD <img src='https://img.shields.io/badge/ACMMM-2025-red'></a> </p>

- Paper: [CGCOD: Class-Guided Camouflaged Object Detection](https://arxiv.org/pdf/2412.18977)

> [!note]
> This work has been accepted by ACMMM2025, and we are about to release all the code. 😊
> 
> Details of the proposed Camoclass dataset can be found in the document for [our dataset](https://github.com/bbdjj/CGCOD/releases/tag/dataset/camoclass.zip).
> Furthermore, we plan to provide more fine-grained textual annotations for the dataset. Regarding the training and testing environments, we aim to investigate the model’s capability to comprehend textual descriptions under complex environmental conditions.

## Dataset Details

To make access easier, we have provided several download links for the images, including Google Drive and a Data Repository. 
However, we do not hold the copyright to these images.
It is your responsibility to review and adhere to the original licenses associated with the images before utilizing them.

The use of these images is entirely at your own risk and discretion. Additionally, please refer to the dataset description below for detailed instructions on how to set up Camoclass before executing our code.

### Prepare Related Data

- **COD10K**: <https://github.com/GewelsJI/SINet-V2>
- **CAMO**: <https://sites.google.com/view/ltnghia/research/camo>
- **NC4K**: <https://github.com/xfflyer/Camouflaged-people-detection>
- **CHAMELEON**: <https://www.polsl.pl/rau6/chameleon-database-animal-camouflage-analysis>

### Data Usage

1. Unzip all the files and set the following path information in a single yaml file `dataset.yaml`:
   - `COD10K`: Directory of **COD10K**, which contains `train` and `test` directories of **COD10K**.
   - `CAMO`: Directory of **CAMO**, which contains `train` and `test` directories of **CAMO**.
   - `NC4K`: Directory of **NC4K**, which contains `img` and `gt` of NC4K.
   - `CHAMELEON`: Directory of **CHAMELEON**, which contains `img` and `gt` of CHAMELEON.
2. Download for the information json for CamoClass dataset (`dataset_info.json`) from 
3. Specify the data roots for training and testing by dataset_info.json.
   
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@article{zhang2024cgcod,
  title={CGCOD: Class-Guided Camouflaged Object Detection},
  author={Zhang, Chenxi and Zhang, Qing and Wu, Jiayun and Pang, Youwei},
  journal={arXiv preprint arXiv:2412.18977},
  year={2024}
}
```
