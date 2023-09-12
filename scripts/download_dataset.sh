FILE=$1

if [ "$FILE" == "StarGAN_CelebA" ]; then
  # Download the imagenet dataset and move validation images to labeled subfolders
  URL="https://huggingface.co/datasets/goodfellowliu/StarGAN_CelebA/resolve/main/celeba.zip"
  ZIP_FILE=./data/celeba.zip
  mkdir -p ./data
  wget -N $URL -O $ZIP_FILE
  unzip $ZIP_FILE -d ./data
  rm $ZIP_FILE
else
  echo "Available arguments are StarGAN_CelebA."
  echo "Example: bash ./scripts/download_dataset.sh StarGAN_CelebA"
  exit 1
fi