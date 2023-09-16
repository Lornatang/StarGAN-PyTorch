FILE=$1

if [ "$FILE" == "StarGAN-CelebA-128x128" ]; then
  URL="https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/StarGAN-CelebA-128x128.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "StarGAN-CelebA-256x256" ]; then
  URL="https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/StarGAN-CelebA-256x256.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "PathDiscriminator-CelebA-128x128" ]; then
  URL="https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/PathDiscriminator-CelebA-128x128.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
elif [ "$FILE" == "PathDiscriminator-CelebA-256x256" ]; then
  URL="https://huggingface.co/goodfellowliu/StarGAN-PyTorch/resolve/main/PathDiscriminator-CelebA-256x256.pth.tar"
  FILE_PATH=./results/pretrained_models
  wget $URL -P $FILE_PATH
else
  echo "Available arguments are StarGAN-CelebA-128x128, StarGAN-CelebA-256x256, PathDiscriminator-CelebA-128x128, PathDiscriminator-CelebA-256x256."
  echo "Example: bash ./scripts/download_weights.sh StarGAN-CelebA-128x128"
  exit 1
fi