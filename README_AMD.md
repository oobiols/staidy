# Download the dataset

gdown https://drive.google.com/uc?id=18Uayn_NrEA8eKhmg4xthO3r1MzDx0LkM

# Move it to the ./datasets_amd directory

mv train.npy ./datasets_amd

# Train, this command will use all available gpus

python amr_amd.py -he 128 -w 512 -ph 16 -pw 16 -nb 4 -bs 4 -lr 1e-4 -mn test --restart 0 --epochs 500



