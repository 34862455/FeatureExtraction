Sign Language Feature Extraction & Fine-Tuning Pipeline
=======================================================

This project provides a pipeline for fine-tuning the S3D model for sign language gloss recognition and extracting features suitable for downstream Sign Language Translation (SLT) using the SignJoey transformer model.

-------------------------------------------------------
Project Structure
-------------------------------------------------------

.
├── s3d_ctc_finetune.py       # Fine-tunes S3D on PHOENIX14T glosses using CTC loss
├── extract_features.py       # Extracts features using a fine-tuned checkpoint
├── model_s3d.py              # Defines the S3D architecture
├── data/                     # Output directory for extracted features (.pt files)
├── checkpoints/              # Directory for saving/loading model checkpoints

-------------------------------------------------------
Step 1: Fine-Tune the S3D Model 
-------------------------------------------------------

To adapt the S3D backbone to the sign language domain, run:

    python s3d_ctc_finetune.py

This script:
- Loads the S3D model pretrained on Kinetics-400,
- Adds a linear gloss classifier,
- Trains using CTC loss with sentence-level gloss annotations from PHOENIX14T,
- Code for partial model freezing commented out
- Code for classifier fine-tuning uncommented
  - NOTE: when only training the classifier, use the secondary (commented out) definition of pickle_features() in extract_features
- Saves checkpoints every epoch as .pt files

-------------------------------------------------------
Step 2: Extract Features Using a Fine-Tuned Checkpoint
-------------------------------------------------------

After fine-tuning completes, use the trained model to extract [T, 1024] features for SignJoey.

Edit `extract_features.py` to point to the desired checkpoint:

    checkpoint = torch.load("checkpoints/finetuning/s3d_classifier_ft_epoch09.pt")

Then run:

    python extract_features.py

This will:
- Load and evaluate the model,
- Pass all PHOENIX14T train/dev/test samples through the S3D encoder,
- Save the resulting feature files as .pt files for each data split

Each file is a gzip-compressed pickle containing a list of dictionaries with:

    {
        "name": str,
        "signer": str,
        "gloss": str,
        "text": str,
        "sign": Tensor  # shape: [T, 1024]
    }

-------------------------------------------------------
Step 3: Train a Recognition Model with SignJoey
-------------------------------------------------------

To use the extracted features in SignJoey:

1. Update the data paths in your config:

    data_path: /your/path/to/project/data
    train: DSG_train.pt
    dev: DSG_dev.pt
    test: DSG_test.pt
    feature_size: 1024

2. Run training:

    python train.py --config configs/SLR.yaml

Ensure your config uses:

    recognition_loss_weight: 1.0
    translation_loss_weight: 0.0
    eval_metric: wer

This setup will train a gloss recognition model on the extracted visual features. The rest of the config is setup according to the hyperparameters presented by Camgoz et al. 2020.
