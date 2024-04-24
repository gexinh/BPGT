# Predicting Genetic Mutation from Whole Slide Images via Biomedical-Linguistic Knowledge Enhanced Multi-label Classification

- - -  
  
# Step 1 Install Packages
Prepare the environment and packages you need as following commands:

    conda env create -f environment.yaml
    pip install -r requirements.txt


# Step 2 PreProcess Dataset
Prepare the dataset from TCGA via the tool 'gdc-client': 

    gdc-client download -m ./PATH_OF_DATA_CART_FROM_TCGA -d ./YOUR_SAVE_PATH 
Then, we can implement the data preprocessing as follows: 
1) masking, segmenting, and slicing patches; [create_patches_fp.py]
2) create cleaning dataset [create_step_2_csv.py]
3) create label space from the cleaned dataset [create_step_3_csv.py]
4) create dataset splits [create_splits.py]
5) extract the knowledge graph [graph_construction.py]
6) extract patch-level features from the pre-trained model [extract_features_fp.py/extract_features_fp.py/] 

The whole pipeline is integrated into the pipeline.py, which we could directly run the following command:

    python [0]_pipeline.py --data tcga --mag 20 -- ori_patch_size 256 --mix_list 012345678

where the '--mix_list' contains the indices of cancer types that you want to use for the dataset construction. 

# Step 3 Run Experiments
    python exp_script --graph_module bpgt --backbone trans_mil --word_embedd_model biobert

this code will call the 'main.py'. In addtion, we could alter the parameters of '--graph_module', '--backbone', '--word_embedd_model' to vary the gene encoder, visual extractor, and linguistic encoding of bpgt respectively.

## Code Structure
    -main.py, pipeline.py, etc
    |-model
    ||-model_exp: details of bpgt with varying graph modules and visual extractors.
    ||-bpgt: the implementation of our gene encoder and label decoder.
    ||-base_model: details of the visual extractor and the gene encoder
    ||-KAT, MCAT, TransMIL, util_layers:
    |
    |-utils: utility functions
    ||-wiki_spider: extract genetic linguistic documents from GeneCard Wiki
    ||-linguistic_encoding.py: extract initial linguistic knowledge embeddings
    ||-graph_utils: construct biomedical encodings for the gene encoder
    |
    |-presets: pre-settings of data preprocessing
    ||-tcga.csv: Pre-settings of TCGA Preprocess
    |
    |-data: 
    ||-splits: examples of data splits, step2, and step3
    ||-dataloader.py: customized WSI dataloader
    ||-extract_biomedical_knowledge.py: utilize to extract gene-pathway mappings
    |
    |-datasets: customized pytorch dataset for WSI
    ||-wsi_dataset.py
    |
    |-vis_utils: utilities of model visualization

## Important Codes
> **./model/base_model.py**:   
it contains details of the modified MIL-backbone utilized for the visual extractor and varying graph aggregation module for the gene encoder. 

> **./utils/graph_utils.py**:  
it contains implementation details for encoding biomedical knowledge 

> **./utils/linguistic_encoding.py**:  
it contains implementation details for encoding linguistic knowledge 

> **./data/extract_biomedical_knowledge.py**:  
it contains implementation details for extracting linguistic knowledge from texture.
