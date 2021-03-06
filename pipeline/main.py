import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import efficientnet_pytorch as efp
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import datasets
import importlib
import utils
from argparse import ArgumentParser
from learning import Learner
from pickle import dump


def init_dataloaders(train_items, val_items, encoder, config, test_items=None):
    module = importlib.import_module(config["AUGMENTATION"]["PY"])
    train_aug = getattr(module, config["AUGMENTATION"]["TRAIN"])()
    val_aug = getattr(module, config["AUGMENTATION"]["VAL"])()
    train_dataset = datasets.TextRecogDataset(
        train_items, encoder, augmentations=train_aug)
    val_dataset = datasets.TextRecogDataset(
        val_items, encoder, augmentations=val_aug)
    if test_items:
        test_dataset = datasets.TextRecogDataset(
            test_items, encoder, augmentations=val_aug)

    train_loader = DataLoader(train_dataset,
                                batch_size=config["BATCH_SIZE"],
                                shuffle=True,
                                num_workers=config["NUM_WORKERS"])

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["BATCH_SIZE"],
                                shuffle=False,
                                num_workers=config["NUM_WORKERS"])
    test_dataloader = None
    if test_items:
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=config["BATCH_SIZE"],
                                     shuffle=False,
                                     num_workers=config["NUM_WORKERS"])
    return {
        "train": train_loader,
        "val": val_dataloader,
        "test": test_dataloader
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_cfg",
                        default="./experiments/baseline.yaml", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.experiment_cfg)
    utils.seed_everything(config["SEED"])

    train_df = pd.read_csv(config["DATA"]["TRAIN"])
    test_df = pd.read_csv(config["DATA"]["TEST"])

    alphabet = [c for clist in train_df["label"].tolist() for c in clist]
    encoder = LabelEncoder()
    encoder.fit(alphabet)

    dump(encoder, config["EXPERIMENT_NAME"]+"_encoder.pkl")

    os.makedirs(config["EXPERIMENT_NAME"], exist_ok=True)
    os.makedirs(f"{config['EXPERIMENT_NAME']}/checkpoints", exist_ok=True)

    train_df, val_df = train_test_split(train_df)

    train_items = train_df.to_dict("record")
    val_items = val_df.to_dict("record")
    test_items = test_df.to_dict("record")

    dataloaders = init_dataloaders(
        train_items, val_items, encoder, config, test_items)
    num_chars = len(set(alphabet))
    
    base = efp.EfficientNet.from_pretrained("efficientnet-b0")

    module = importlib.import_module(config["MODEL"]["PY"])
    model = getattr(module, config["MODEL"]["ARCH"])(base, num_chars)

    if len(config["RESUME"]) > 0:
        ckpt_path = config["RESUME"]
        ckpt_dict = torch.load(ckpt_path)
        prev_state = utils.load_pytorch_model(ckpt_dict['state_dict'])
        model.load_state_dict(prev_state)

    module = importlib.import_module(config["OPTIMIZER"]["PY"])
    optimizer = getattr(module, config["OPTIMIZER"]["CLASS"])(
        model.parameters(), **config["OPTIMIZER"]["ARGS"])

    module = importlib.import_module(config["SCHEDULER"]["PY"])
    scheduler = getattr(module, config["SCHEDULER"]["CLASS"])(
        optimizer, **config["SCHEDULER"]["ARGS"])

    early_stop_callback = EarlyStopping(
        **config["EARLY_STOPPING"]["ARGS"]
    )

    checkpoint_callback = ModelCheckpoint(
        **config["CHECKPOINT"]["ARGS"]
    )
    logger = TensorBoardLogger(
        config["EXPERIMENT_NAME"], name=config["EXPERIMENT_NAME"])

    lightning_model = Learner(
        dataloaders=dataloaders,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        encoder=encoder
    )
    # grad_clip = config["GRADIENT_CLIPPING"]
    # grad_acum = config["GRADIENT_ACCUMULATION_STEPS"]
    trainer = pl.Trainer(gpus=config["GPUS"],
                            max_epochs=config["EPOCHS"],
                            num_sanity_val_steps=0,
                            logger=logger,
                            #   val_percent_check=0.1,
                            #   train_percent_check=0.1,
                            # gradient_clip_val=grad_clip,
                            # accumulate_grad_batches=grad_acum,
                            early_stop_callback=early_stop_callback,
                            checkpoint_callback=checkpoint_callback)
    trainer.fit(lightning_model)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    main()
