export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="bert-trainer"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-v100,count=1" \
  --metadata="install-nvidia-driver=True"
