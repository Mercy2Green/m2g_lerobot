# cd ~/vla/src/pyorbbecsdk
# export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
# sudo bash ./scripts/install_udev_rules.sh
# sudo udevadm control --reload-rules
# sudo udevadm trigger
# cd /home/hpx/peter_ws/m2g_lerobot/lerobot


# conda activate ur
cd /home/hpx/peter_ws/m2g_lerobot/lerobot
DATA_SAVE_DIR="m2g"

rm -r /media/hpx/newMemery/datasets/${DATA_SAVE_DIR}

export PYTHONPATH="/home/hpx/peter_ws/m2g_lerobot:$PYTHONPATH"

python /home/hpx/peter_ws/m2g_lerobot/lerobot/record_m2g.py \
  --srobot.type=koch_follower \
  --srobot.port=/ttyUSB0 \
  --teleop.type=koch_leader \
  --teleop.port=/ttyUSB1 \
  --dataset.repo_id=hpx/cube1 \
  --dataset.root=/media/hpx/newMemery/datasets/${DATA_SAVE_DIR} \
  --dataset.push_to_hub=False \
  --dataset.num_episodes=100 \
  --dataset.single_task="Grab the toy and put it in the box" \
  --auto_mode=True \

