# CGCOD

- Paper: [CGCOD: Class-Guided Camouflaged Object Detection](https://arxiv.org/pdf/2412.18977)


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
2. Download for the information json for CamoClass dataset (`class_info.json` and `sample_info.json`) from 
3. Specify the data roots for training and testing by class_info.json.

## LICENSE of 
